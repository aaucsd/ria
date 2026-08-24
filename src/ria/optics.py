"""The forward optics, stated explicitly.

Every geometric quantity is an argument. Nothing is assumed, nothing is filled
in from a reference prototype — if a number affects the image, you supply it.

The layout along the optical axis:

        sensor plane            lens plane              scintillator
             │                       │              ┌───────────────────┐
             │◄─── imaging_distance ───────────►│◄── working_distance ─┤ z=0          z=Lz │
             │                       │              └───────────────────┘
                                  lens i at (xᵢ, yᵢ),
                                  focal length fᵢ, aperture radius aᵢ

A source at `(X, Y, z)` — lateral position on the scintillator axes, `z`
measured into the volume from its front face — sits a distance
`u = working_distance + z` in front of the lens plane. Each lens then paints one
Gaussian blob on the sensor:

    centre   sᵢ = (xᵢ, yᵢ) + (imaging_distance/u)·((X, Y) − (xᵢ, yᵢ))
    image    dᵢ = fᵢ·u / (u − fᵢ)
    blur     σᵢ = (aᵢ/2)·|imaging_distance − dᵢ| / |dᵢ| + blur_floor
    weight   wᵢ ∝ aᵢ² / u²          (solid angle the aperture subtends)

Blobs are integrated exactly over each pixel, and the total is normalised to
the detected photon count.
"""
import numpy as np


def _erf(x):
    """Vectorised error function (Abramowitz & Stegun 7.1.26, ~1e-7)."""
    x = np.asarray(x, float)
    s = np.sign(x); ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return s * y


class LensPlane:
    """An array of thin lenses. Every lens is stated individually.

    lenses      a list of records, one per lens:
                    {"position": (x, y),   centre on the lens plane, mm
                     "f": focal_length,    mm
                     "aperture": radius}   aperture RADIUS, mm
    blur_floor  residual blur added to every blob, mm (finite optics,
                diffraction, sensor MTF — the σ₀ of the design model)
    """

    def __init__(self, lenses, blur_floor):
        if not lenses:
            raise ValueError("lenses: supply at least one lens record")
        pos, f, ap = [], [], []
        for i, L in enumerate(lenses):
            missing = {"position", "f", "aperture"} - set(L)
            if missing:
                raise ValueError(
                    f"lens {i}: missing {sorted(missing)}; each lens is "
                    f'{{"position": (x, y), "f": mm, "aperture": mm}}')
            x, y = L["position"]
            pos.append((float(x), float(y)))
            f.append(float(L["f"]))
            ap.append(float(L["aperture"]))
        self.positions = np.array(pos, float)
        self.focal_lengths = np.array(f, float)
        self.apertures = np.array(ap, float)
        self.blur_floor = float(blur_floor)

    @property
    def n(self):
        return len(self.positions)

    def min_separation(self):
        if self.n < 2:
            return float("inf")
        d = np.linalg.norm(self.positions[:, None, :] - self.positions[None, :, :],
                           axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min())

    def __repr__(self):
        f = self.focal_lengths
        return (f"LensPlane(n={self.n}, f∈[{f.min():.2f},{f.max():.2f}]mm, "
                f"aperture∈[{self.apertures.min():.2f},{self.apertures.max():.2f}]mm, "
                f"blur_floor={self.blur_floor}mm)")


class SensorPlane:
    """The pixellated detector.

    size    (width, height) of the active area, mm
    pixels  (nx, ny) pixel counts across that area
    imaging_distance   distance from the lens plane to the sensor, mm
    dark_count  per-pixel dark rate added to every exposure
    """

    def __init__(self, size, pixels, imaging_distance, dark_count):
        self.width, self.height = (float(v) for v in size)
        self.nx, self.ny = (int(v) for v in pixels)
        self.imaging_distance = float(imaging_distance)
        self.dark_count = float(dark_count)

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def pixel_size(self):
        return (self.width / self.nx, self.height / self.ny)

    def edges(self):
        return (np.linspace(-self.width / 2, self.width / 2, self.nx + 1),
                np.linspace(-self.height / 2, self.height / 2, self.ny + 1))

    def __repr__(self):
        px, py = self.pixel_size
        return (f"SensorPlane({self.width}×{self.height}mm, "
                f"{self.nx}×{self.ny} px of {px:.3f}×{py:.3f}mm, "
                f"imaging_distance={self.imaging_distance}mm, dark={self.dark_count:g})")


def blob_geometry(lenses, sensor, position, working_distance):
    """Per-lens blob centre, blur and weight for a source at `position`.

    Returns (centres (N,2), sigma (N,), weight (N,), on_sensor (N,) bool).
    """
    X, Y, z = (float(v) for v in position)
    u = working_distance + z                                   # source → lens plane
    if u <= 0:
        raise ValueError(f"source is not in front of the lens plane (u={u} mm)")
    p = lenses.positions
    m = sensor.imaging_distance / u                               # magnification
    centres = p + m * (np.array([X, Y]) - p)

    f = lenses.focal_lengths
    with np.errstate(divide="ignore", invalid="ignore"):
        d = f * u / (u - f)                            # image distance
        sigma = (lenses.apertures / 2.0) * np.abs(sensor.imaging_distance - d) / np.abs(d)
    sigma = np.where(np.isfinite(sigma), sigma, np.inf) + lenses.blur_floor

    weight = (lenses.apertures ** 2) / (u ** 2)        # solid angle ∝ a²/u²
    on = ((np.abs(centres[:, 0]) <= sensor.width / 2) &
          (np.abs(centres[:, 1]) <= sensor.height / 2) &
          np.isfinite(sigma))
    return centres, sigma, weight, on


def image_rate(lenses, sensor, position, photons, working_distance):
    """Expected photon count per pixel.

    CANONICAL MODEL (used identically by simulation and inversion): each lens
    paints a Gaussian blob; a pixel's expected count is the blob DENSITY at the
    pixel centre, with the pixel's own extent folded in as extra variance
    (var = sigma^2 + w^2/12 per axis — the standard second-moment convolution),
    times the pixel area. Everything is exp/sqrt/ratio arithmetic, so the
    likelihood can be written EXACTLY in the solver's language — no erf, no
    approximation gap between forward and inverse.

    The pattern is normalised over the sensor (sum P = 1), so
    `photons` is alpha — the expected DETECTED count — plus dark floor.
    """
    prof = pattern_profile(lenses, sensor, position, working_distance)
    tot = prof.sum()
    rate = np.full(sensor.shape, sensor.dark_count)
    if tot > 0:
        rate = rate + prof * (float(photons) / tot)
    return rate


def pattern_profile(lenses, sensor, position, working_distance):
    """The UNNORMALISED pattern: per-pixel sum over lenses of
    area * (a^2/(2 pi sqrt(vx vy) u^2)) * exp(-(dx^2/2vx + dy^2/2vy)).
    This exact expression is what the emitted model contains, term for term."""
    centres, sigma, weight, on = blob_geometry(lenses, sensor, position,
                                               working_distance)
    prof = np.zeros(sensor.shape)
    if not on.any():
        return prof
    ex, ey = sensor.edges()
    cxp = 0.5 * (ex[:-1] + ex[1:]); cyp = 0.5 * (ey[:-1] + ey[1:])
    wx2 = (ex[1] - ex[0]) ** 2 / 12.0; wy2 = (ey[1] - ey[0]) ** 2 / 12.0
    area = (ex[1] - ex[0]) * (ey[1] - ey[0])
    for c, s, w in zip(centres[on], sigma[on], weight[on]):
        vx, vy = s * s + wx2, s * s + wy2
        gx = np.exp(-(cxp - c[0]) ** 2 / (2 * vx))
        gy = np.exp(-(cyp - c[1]) ** 2 / (2 * vy))
        prof += (w * area / (2 * np.pi * np.sqrt(vx * vy))) * np.outer(gy, gx)
    return prof


def n_visible(lenses, sensor, position, working_distance):
    """How many lenses land their blob on the sensor for this source."""
    return int(blob_geometry(lenses, sensor, position, working_distance)[3].sum())


# ── layout helpers — call them explicitly, they are not defaults ──────────
def hex_layout(rings, pitch):
    """Hex-packed lens centres: the central lens plus `rings` surrounding
    rings at spacing `pitch` (mm). 1 ring → 7 lenses, 2 → 19, 3 → 37."""
    pts = [(0.0, 0.0)]
    for r in range(1, int(rings) + 1):
        for k in range(6 * r):
            a = 2 * np.pi * k / (6 * r)
            pts.append((r * pitch * np.cos(a), r * pitch * np.sin(a)))
    return np.array(pts)


def grid_layout(nx, ny, pitch):
    """Rectangular lens centres, `nx × ny` at spacing `pitch` (mm)."""
    xs = (np.arange(nx) - (nx - 1) / 2) * pitch
    ys = (np.arange(ny) - (ny - 1) / 2) * pitch
    return np.array([(x, y) for y in ys for x in xs])
