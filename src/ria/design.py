"""Lens-array design: how precisely can a given array locate a source, and
which array locates it best?

This is the *design* side of the instrument (the companion to the inversion in `system.py`/`model_export.py`,
which is the *measurement* side). It implements the chief-ray model and the
Fisher/Cramér–Rao analysis of the lens-design write-up:

  blob centre    sx = xi + (S*/z)(r − xi),      sy = yi (1 − S*/z)
  image distance di = fi z / (z − fi)
  blur radius    σi(z) = a |S* − di| / |di| + σ0,   a = a_lens/2, σ0 = 0.12 mm
  depth Fisher   Iz(r,z) = (Nγ/N) Σ_i 1/σi²  [ (a S*/z²)² + (S*/z²)²((r−xi)² + yi²) ]
  depth CRB      CRBz(r,z) = 1 / sqrt(Iz(r,z))          (mm)

Depth enters *only* through the magnification S*/z — that is how a flat array
encodes depth at all. A lens contributes most when it is sharp at that depth
(small σi) and when it sits far off-axis from the source (the lever arm).

The design problem is then, over the working grid G:

  minimise  (1/|G|) Σ_{(r,z) ∈ G} log CRBz(r,z)
  subject to  ‖(xi,yi) − (xj,yj)‖ ≥ 3.5 mm,  fi ∈ [3, 16] mm,
              every lens inside the mounting area.
"""
import numpy as np

# ── fixed system parameters (from the write-up) ───────────────────────────
S_STAR = 20.477      # mm   sensor ← lens-array distance
SENSOR_W = 46.15     # mm   sensor width
SENSOR_H = 32.84     # mm   sensor height
A_LENS = 1.5         # mm   lens radius
SIGMA0 = 0.12        # mm   residual blur floor
F_MIN, F_MAX = 3.0, 16.0
MIN_SEP = 3.5        # mm   lens-to-lens clearance
N_PHOTONS = 100      # detected photons shared across the array


class LensArray:
    """A lens array as a plain list of records:

        [{"position": (x, y), "f": focal_length, "aperture": radius}, ...]

    plus the system-wide blur_floor, imaging_distance and sensor_size. Every
    optical quantity is supplied — nothing is assumed."""

    def __init__(self, lenses, blur_floor, imaging_distance, sensor_size):
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
        self.pos = np.array(pos, float)
        self.f = np.array(f, float)
        self.apertures = np.array(ap, float)
        self.blur_floor = float(blur_floor)
        self.imaging_distance = float(imaging_distance)
        self.sensor_size = tuple(float(v) for v in sensor_size)

    # ---- constructors ----------------------------------------------------
    @classmethod
    def hex(cls, rings, pitch, focal_set, apertures, blur_floor, imaging_distance,
            sensor_size):
        """Hex-packed array: central lens plus `rings` rings at `pitch`,
        focal lengths cycled over `focal_set`."""
        pts = [(0.0, 0.0)]
        for r in range(1, int(rings) + 1):
            for k in range(6 * r):
                a = 2 * np.pi * k / (6 * r)
                pts.append((r * pitch * np.cos(a), r * pitch * np.sin(a)))
        recs = [{"position": p, "f": focal_set[i % len(focal_set)],
                 "aperture": apertures} for i, p in enumerate(pts)]
        return cls(recs, blur_floor, imaging_distance, sensor_size)

    @classmethod
    def _legacy_default(cls, n=42, pitch=3.6, focal_set=(3, 4.5, 6, 7.5, 9, 12, 15)):
        """Hex-packed array with focal lengths cycled over `focal_set`, i.e.
        the multi-focal design: each lens is a vote for one depth."""
        pts, ring = [(0.0, 0.0)], 1
        while len(pts) < n:
            for k in range(6 * ring):
                ang = 2 * np.pi * k / (6 * ring)
                pts.append((ring * pitch * np.cos(ang), ring * pitch * np.sin(ang)))
            ring += 1
        pts = pts[:n]
        f = [focal_set[i % len(focal_set)] for i in range(n)]
        return cls(pts, f)

    @property
    def n(self):
        return len(self.f)

    # ---- constraint checking --------------------------------------------
    def violations(self, min_sep=MIN_SEP, f_range=(F_MIN, F_MAX),
                   sensor=None):
        """Which design constraints this array breaks (empty ⇒ feasible)."""
        out = []
        d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() < min_sep - 1e-9:
            out.append(f"lens spacing {d.min():.2f} mm < {min_sep} mm")
        fmin, fmax = f_range
        if self.f.min() < fmin - 1e-9 or self.f.max() > fmax + 1e-9:
            out.append(f"focal lengths [{self.f.min():.2f}, {self.f.max():.2f}] "
                       f"outside [{fmin}, {fmax}] mm")
        w, h = sensor or self.sensor_size
        if np.any(np.abs(self.pos[:, 0]) > w / 2) or \
           np.any(np.abs(self.pos[:, 1]) > h / 2):
            out.append("a lens sits outside the mounting area")
        return out

    def __repr__(self):
        return (f"LensArray(n={self.n}, f∈[{self.f.min():.1f},{self.f.max():.1f}]mm, "
                f"{len(set(np.round(self.f,3)))} distinct focal lengths)")


# ── the chief-ray forward model ───────────────────────────────────────────
def blob_centres(array, r, z):
    """Where each lens lands its blob on the sensor, for a source at lateral
    radius `r` and depth `z`.  (Source taken on the +x axis, as in the grid.)"""
    m = array.imaging_distance / z
    sx = array.pos[:, 0] + m * (r - array.pos[:, 0])
    sy = array.pos[:, 1] * (1.0 - m)
    return np.stack([sx, sy], axis=1)


def blur_radii(array, z):
    """Defocus blur σi(z) per lens (mm).  A lens is sharp only where its image
    distance matches S*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        di = array.f * z / (z - array.f)
        sig = ((array.apertures / 2.0) * np.abs(array.imaging_distance - di) / np.abs(di)
               + array.blur_floor)
    return np.where(np.isfinite(sig), sig, np.inf)


def on_sensor(centres, sensor):
    w, h = sensor
    return (np.abs(centres[:, 0]) <= w / 2) & (np.abs(centres[:, 1]) <= h / 2)


# ── Fisher information and the Cramér–Rao bound on depth ──────────────────
def depth_fisher(array, r, z, photons=N_PHOTONS):
    """Fisher information about depth z for a source at (r, z)."""
    c = blob_centres(array, r, z)
    keep = on_sensor(c, array.sensor_size)
    if not keep.any():
        return 0.0
    sig = blur_radii(array, z)[keep]
    x, y = array.pos[keep, 0], array.pos[keep, 1]
    a = array.apertures[keep] / 2.0
    T = array.imaging_distance
    lever = (r - x) ** 2 + y ** 2
    term = ((a * T / z ** 2) ** 2 + (T / z ** 2) ** 2 * lever) / sig ** 2
    return float(photons / array.n * np.sum(term))


def depth_crb(array, r, z, photons=N_PHOTONS):
    """Cramér–Rao lower bound on depth precision (mm) — no unbiased estimator
    can do better than this standard deviation."""
    I = depth_fisher(array, r, z, photons)
    return float("inf") if I <= 0 else 1.0 / np.sqrt(I)


def crb_map(array, radii=(5, 10, 15, 20, 25), depths=tuple(range(10, 51, 5)),
            photons=N_PHOTONS):
    """CRB on depth across the working volume → (radii, depths, CRB[r, z])."""
    M = np.array([[depth_crb(array, r, z, photons) for z in depths]
                  for r in radii])
    return np.asarray(radii, float), np.asarray(depths, float), M


def design_score(array, radii=(5, 10, 15, 20, 25),
                 depths=tuple(range(10, 51, 5)), photons=N_PHOTONS):
    """The design objective: mean log CRB over the working grid (lower is
    better). Equivalent to the geometric-mean depth precision."""
    _, _, M = crb_map(array, radii, depths, photons)
    finite = np.isfinite(M) & (M > 0)
    if not finite.any():
        return float("inf")
    return float(np.mean(np.log(M[finite])))


# ── the design problem ────────────────────────────────────────────────────
def optimise_focal_lengths(array, radii, depths, photons, f_range=(F_MIN, F_MAX),
                           rounds=6, seed=0, verbose=False):
    """Choose focal lengths to minimise mean log CRB over the working volume,
    holding lens positions fixed (so the spacing and mounting constraints hold
    by construction; only fi ∈ [F_MIN, F_MAX] is active).

    Coordinate descent: repeatedly re-optimise one lens's focal length on a
    fine 1-D sweep with the others fixed. The objective separates additively
    over lenses at fixed positions, so this converges quickly and is exact per
    coordinate — no gradients, no local-minimum drama.
    """
    best = LensArray([{"position": tuple(p), "f": float(f), "aperture": float(a)}
                      for p, f, a in zip(array.pos, array.f, array.apertures)],
                     array.blur_floor, array.imaging_distance,
                     array.sensor_size)
    score = design_score(best, radii, depths, photons)
    grid = np.linspace(f_range[0], f_range[1], 131)
    rng = np.random.default_rng(seed)
    for it in range(rounds):
        improved = False
        for i in rng.permutation(best.n):
            f0, cur = best.f[i], score
            for cand in grid:
                best.f[i] = cand
                s = design_score(best, radii, depths, photons)
                if s < cur - 1e-12:
                    cur, f0 = s, cand
            best.f[i] = f0
            if cur < score - 1e-12:
                score, improved = cur, True
        if verbose:
            print(f"  round {it+1}: mean log CRB = {score:.4f}")
        if not improved:
            break
    return best, score


# ── the design problem as a REKIN QUERY (2026-08-04) ──────────────────────
# The same optimisation, handed to the global solver instead of coordinate
# descent: decision variables are the focal lengths f_i ∈ [F_MIN, F_MAX]
# (positions fixed, so spacing/mounting hold by construction).  Everything
# f-independent — the on-sensor mask, the lever arms, the z-scalings — is
# precomputed into constants; the .rn contains only the true unknowns.
#
#   objective "mean":     minimise  Σ_G −log I_z(r,z)
#                         (= 2|G| · mean log CRB + const, same argmin)
#   objective "minimax":  minimise t  s.t.  t ≥ −½ log I_z(r,z)  ∀(r,z)∈G
#                         (t is the worst-case log CRB over the volume —
#                          a query the sweep optimiser has no analogue for)
#
# σ_i(z) uses the same smoothed |·| as the imaging model
# (sqrt(t²+1e-12) ≈ |t| to ≤1e-6), so the emitted model matches
# `blur_radii`/`depth_fisher` to float round-off away from the kink.
def emit_design_model(array, radii, depths, photons, f_range=(F_MIN, F_MAX),
                      objective="mean", path=None):
    """Write the lens-design problem as a macro-structured .rn over exactly
    the focal lengths (plus `t` for minimax).  Returns (path, grid) where
    grid is the list of included (r, z) points (nonempty on-sensor mask —
    the same finite-CRB mask design_score uses)."""
    import os
    import tempfile
    g = lambda v: "%.10g" % v
    n = array.n
    T = array.imaging_distance
    grid = []
    for r in radii:
        for z in depths:
            m = T / z
            sx = array.pos[:, 0] + m * (r - array.pos[:, 0])
            sy = array.pos[:, 1] * (1.0 - m)
            keep = (np.abs(sx) <= array.sensor_size[0] / 2) & \
                   (np.abs(sy) <= array.sensor_size[1] / 2)
            if keep.any():
                grid.append((float(r), float(z), keep))
    L = ["# Lens-array design as a Rekin query (ria."
         "design).",
         f"# {n} focal lengths in [{g(f_range[0])}, {g(f_range[1])}] mm; "
         f"{len(grid)} working-grid points;",
         f"# objective = {objective} of the depth CRB [D Sec. 4].  Positions "
         "fixed, so the",
         "# spacing/mounting constraints of [D Sec. 2] hold by construction.",
         "Variables:"]
    for i in range(n):
        L.append(f"  Real f{i} : [{g(f_range[0])}, {g(f_range[1])}]")
    if objective == "minimax":
        L.append("  Real t : [-15, 15]")
    L.append("")
    zs = sorted({z for _, z, _ in grid})
    L.append("# per-lens, per-depth defocus blur sigma_i(z) "
             "[D Sec. 3(b)], smoothed |.|")
    for z in zs:
        zi = int(round(z))
        for i in range(n):
            a2 = array.apertures[i] / 2.0
            L.append(
                f"Define sig_{i}_{zi} := ({g(a2)}*sqrt(({g(T)}*({g(z)} - f{i})"
                f"/(f{i}*{g(z)}) - 1)^2 + 1e-12) + {g(array.blur_floor)})")
    L.append("# depth Fisher information per working-grid point "
             "(constants precomputed)")
    inames = []
    for (r, z, keep) in grid:
        zi = int(round(z))
        terms = []
        for i in range(n):
            if not keep[i]:
                continue
            a2 = array.apertures[i] / 2.0
            lever = (r - array.pos[i, 0]) ** 2 + array.pos[i, 1] ** 2
            C = (photons / n) * ((a2 * T / z ** 2) ** 2 +
                                 (T / z ** 2) ** 2 * lever)
            terms.append(f"{g(C)}/(sig_{i}_{zi}*sig_{i}_{zi})")
        nm = f"I_{int(round(r))}_{zi}"
        inames.append(nm)
        L.append(f"Define {nm} := (" + " + ".join(terms) + ")")
    if objective == "minimax":
        L += ["", "Constraints:",
              "  # t dominates the log CRB of every working-grid point"]
        for nm in inames:
            L.append(f"  t + 0.5*log({nm}) >= 0 ~ 1e-3")
        L += ["", "Minimize:", "  t ~ 1e-3"]
    else:
        L += ["", "Minimize:",
              "  # sum of -log I over the grid = 2|G| * mean log CRB + const",
              "  " + " ".join(f"- log({nm})" for nm in inames) + " ~ 1e-3"]
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".rn")
        os.close(fd)
    open(path, "w").write("\n".join(L) + "\n")
    return path, [(r, z) for r, z, _ in grid]


def optimise_focal_lengths_rekin(array, radii, depths, photons,
                                 f_range=(F_MIN, F_MAX), objective="mean",
                                 budget=120.0, keep_model=None, stall=0.0):
    """Solve the lens-design problem with Rekin's global stack.  Returns
    (LensArray, solver status, solver objective).  For objective="mean" the
    reported design score is `design_score(best)`; for "minimax" the solver
    objective IS the worst-case log CRB t*.  `stall=N` returns N seconds after
    the last incumbent improvement (0 = run the full budget)."""
    import os
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.setrecursionlimit(1000000)
    from rekin import solve as solve_file
    path, grid = emit_design_model(array, radii, depths, photons,
                                   f_range=f_range, objective=objective,
                                   path=keep_model)
    try:
        r = solve_file(path, budget=budget, stall_stop=stall)
    finally:
        if keep_model is None:
            os.unlink(path)
    sol = r.get("sol") or {}
    best = LensArray(
        [{"position": tuple(p), "f": float(sol.get(f"f{i}", array.f[i])),
          "aperture": float(a)}
         for i, (p, a) in enumerate(zip(array.pos, array.apertures))],
        array.blur_floor, array.imaging_distance, array.sensor_size)
    return best, r["status_name"], float(r["obj"])
