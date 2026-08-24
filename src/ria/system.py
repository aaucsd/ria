"""Build an imaging system, simulate it, invert it.

Every physical quantity is supplied by you. There are no default geometries:
if a number changes the image, the API asks for it.

    import ria

    system = ria.imager()
    system.addScint(size=(30, 30, 40), working_distance=5.0)
    system.addLens(positions=ria.hex_layout(rings=3, pitch=3.6),
                   focal_lengths=[...], apertures=1.5, blur_floor=0.12)
    system.addDetector(size=(46.15, 32.84), pixels=(32, 32),
                       imaging_distance=20.477, dark_count=1e-9)

    s1    = system.addVertex([4, 2, 30], photons=150)
    image = system.forward(s1, seed=1)
    r     = system.invert(image, sources=1, truth=s1)
"""
import numpy as np

from .optics import (LensPlane, SensorPlane, image_rate, n_visible,
                     hex_layout, grid_layout)


class Scintillator:
    """The active volume.

    size      overall (x, y, z) extent in mm. x and y are centred on the
              optical axis; z runs from the face nearest the lenses into the
              volume, so z ∈ (0, Lz].
    working_distance  distance from the lens plane to that front face, mm.
    """

    def __init__(self, size, working_distance):
        self.size = tuple(float(v) for v in size)
        self.working_distance = float(working_distance)

    @property
    def bounds(self):
        Lx, Ly, Lz = self.size
        return np.array([[-Lx / 2, Lx / 2], [-Ly / 2, Ly / 2], [1e-6, Lz]])

    def contains(self, p):
        b = self.bounds
        p = np.asarray(p, float)
        return bool(np.all((p >= b[:, 0]) & (p <= b[:, 1])))

    def __repr__(self):
        return f"Scintillator(size={self.size}mm, working_distance={self.working_distance}mm)"


class Vertex:
    """An interaction: where it happened, and how many photons it delivered to
    the sensor."""

    def __init__(self, position, photons, name=None):
        self.position = tuple(float(v) for v in position)
        self.photons = float(photons)
        self.name = name

    def __repr__(self):
        n = f"{self.name}: " if self.name else ""
        return f"Vertex({n}{self.position}mm, {self.photons:g} photons)"


class Inversion:
    """A recovered vertex: position, fitted photon count, likelihood value,
    and the solver's status for the run that produced it."""

    def __init__(self, position, photons, nll, truth=None, status=None):
        self.position = tuple(float(v) for v in position)   # (x, y, z) mm
        self.photons = float(photons)      # fitted alpha (detected photons)
        self.nll = float(nll)              # Bernoulli NLL at the fit
        self.status = status               # delta-global | local | best-effort
        self.truth = None if truth is None else tuple(float(v) for v in truth)

    @property
    def error(self):
        """Per-axis |error|, when a truth was supplied."""
        if self.truth is None:
            return None
        return tuple(abs(a - b) for a, b in zip(self.position, self.truth))

    def __repr__(self):
        p = ", ".join(f"{v:.3f}" for v in self.position)
        e = "" if self.truth is None else \
            f", error={float(np.linalg.norm(self.error)):.3f}mm"
        s = "" if self.status is None else f", status={self.status}"
        return f"Inversion(position=({p})mm, photons={self.photons:.1f}{e}{s})"


class _Model:
    """Adapter presenting the system to the fitting code."""

    def __init__(self, lenses, sensor, scint):
        self.lenses, self.sensor, self.scint = lenses, sensor, scint
        self.dark_count = sensor.dark_count
        self.sensor_pixels = sensor.nx
        self.bounds = scint.bounds

    def rate(self, position, photons):
        return image_rate(self.lenses, self.sensor, position, photons,
                          self.scint.working_distance)


class Imager:
    """A plenoptic imaging system assembled from explicit components."""

    def __init__(self):
        self.scint = None
        self.lenses = None
        self.sensor = None
        self.vertices = []
        self._model = None

    # ---- components ------------------------------------------------------
    def addScint(self, size, working_distance):
        """Active volume: `size` = overall (x, y, z) mm; `working_distance` = distance
        from the lens plane to the front face, mm."""
        self.scint = Scintillator(size, working_distance)
        self._model = None
        return self.scint

    def addLens(self, lenses, blur_floor):
        """The lens array: a plain list with one record per lens.

            lenses = [{"position": (0.0, 0.0), "f": 9.0, "aperture": 1.5},
                      {"position": (3.6, 0.0), "f": 12.0, "aperture": 1.5},
                      ...]

        blur_floor is the residual blur added to every blob, mm.
        """
        self.lenses = LensPlane(lenses, blur_floor)
        self._model = None
        return self.lenses

    def addDetector(self, size, pixels, imaging_distance, dark_count):
        """The sensor: `size` = (width, height) mm; `pixels` = (nx, ny);
        `imaging_distance` = lens-plane-to-sensor distance, mm; `dark_count` per pixel."""
        self.sensor = SensorPlane(size, pixels, imaging_distance, dark_count)
        self._model = None
        return self.sensor

    def addVertex(self, position, photons, name=None):
        """An interaction at `position` (mm) delivering `photons` detected
        photons."""
        self._require(scint=True)
        if not self.scint.contains(position):
            raise ValueError(
                f"vertex {tuple(position)} is outside the scintillator "
                f"{self.scint.size}mm; bounds are {self.scint.bounds.tolist()}")
        v = Vertex(position, photons, name)
        self.vertices.append(v)
        return v

    # ---- completeness ----------------------------------------------------
    def _require(self, scint=False, lenses=False, sensor=False):
        missing = []
        if scint and self.scint is None:
            missing.append("addScint(size=..., working_distance=...)")
        if lenses and self.lenses is None:
            missing.append("addLens(lenses=[...], blur_floor=...)")
        if sensor and self.sensor is None:
            missing.append("addDetector(size=..., pixels=..., imaging_distance=..., "
                           "dark_count=...)")
        if missing:
            raise ValueError("system incomplete; call " + " and ".join(missing))

    def _build(self):
        self._require(scint=True, lenses=True, sensor=True)
        if self._model is None:
            self._model = _Model(self.lenses, self.sensor, self.scint)
        return self._model

    def visibility(self, position):
        """Number of lenses landing a blob on the sensor for a source at
        `position` — 0 means the point cannot be measured at all."""
        self._require(scint=True, lenses=True, sensor=True)
        return n_visible(self.lenses, self.sensor, position, self.scint.working_distance)

    def precision_floor(self, position, photons):
        """The Cramer-Rao bound on depth for this system at `position`, in mm
        [D Sec. 4] — the best standard deviation ANY unbiased estimator could
        achieve, set by the optics and photon budget alone.

        Use it to interpret a fit:

          * reported precision ~ this floor  → you are information-limited;
            the only way to do better is a better instrument or more photons.
          * reported precision >> this floor → the search is under-resolved;
            raise `grid` / `refine` in invert().
          * error >> reported precision      → the fit is in the WRONG BASIN.
            It is confidently wrong, not imprecise; try `polish=True`, more
            `candidates`, or an `roi`.

        Caveat: the bound assumes Poisson photon COUNTING [D Sec. 4], while
        this detector is binary 1-bit [P Sec. II]. A 1-bit pixel discards
        multiplicity, so the achievable precision is strictly worse than this
        floor — treat it as optimistic.
        """
        from .design import LensArray, depth_crb
        self._require(scint=True, lenses=True, sensor=True)
        x, y, z = (float(v) for v in position)
        arr = LensArray(
            lenses=[{"position": tuple(p), "f": float(f), "aperture": float(a)}
                    for p, f, a in zip(self.lenses.positions,
                                       self.lenses.focal_lengths,
                                       self.lenses.apertures)],
            blur_floor=self.lenses.blur_floor,
            imaging_distance=self.sensor.imaging_distance,
            sensor_size=(self.sensor.width, self.sensor.height))
        # design-side z is measured from the LENS PLANE; system-side z is depth
        # into the scintillator, so add the working distance.
        return depth_crb(arr, r=float(np.hypot(x, y)),
                         z=self.scint.working_distance + z, photons=photons)

    # ---- forward ---------------------------------------------------------
    def rate(self, vertex=None):
        """Expected photons per pixel (noiseless), summing the given
        vertices."""
        m = self._build()
        out = None
        for v in self._vertex_list(vertex):
            r = m.rate(v.position, v.photons)
            out = r if out is None else out + (r - m.dark_count)
        return out

    def forward(self, vertex=None, seed=None, binary=True):
        """Simulate a measurement. `binary=True` gives one bit per pixel
        (Bernoulli 1−e^{−λ}) and requires `seed`; `binary=False` gives the
        noiseless expected counts."""
        rate = self.rate(vertex)
        m = self._build()
        if float(rate.sum() - m.dark_count * rate.size) <= 0:
            vs = self._vertex_list(vertex)
            raise ValueError(
                "no signal: no lens lands a blob on the sensor for "
                f"{[v.position for v in vs]}. Check system.visibility(p).")
        if not binary:
            return rate
        if seed is None:
            raise ValueError("forward(..., seed=n) is required for a binary "
                             "draw; pass binary=False for the noiseless rate")
        rng = np.random.default_rng(seed)
        return (rng.random(rate.shape) < (1.0 - np.exp(-rate))).astype(np.uint8)

    def _vertex_list(self, vertex):
        vs = (self.vertices if vertex is None else
              (list(vertex) if isinstance(vertex, (list, tuple)) else [vertex]))
        if not vs:
            raise ValueError("no vertices; call addVertex(...) first")
        return vs

    # ---- validate: independent forward-pass cross-check ------------------
    def validate(self, inversion, image, truth=None, verbose=True):
        """Cross-check an inversion against an INDEPENDENT numpy forward
        pass — no solver code involved.  Recomputes the exact-model NLL
        at the recovered vertices and reports:

          nll_solver     the NLL Rekin reported
          nll_numpy      the same likelihood recomputed through the numpy
                         forward optics at the recovered vertices
          model_rel_err  |nll_solver - nll_numpy| / (1+|nll_numpy|) — how
                         faithfully the emitted model matches the physics
          nll_truth      numpy NLL at the ground truth (when given) with
                         its true photon counts
          beats_truth    nll_numpy <= nll_truth: the solver's point explains
                         the image at least as well as the truth does

        `inversion` is what `invert()` returned (one Inversion or a list);
        `truth` is a Vertex/list of Vertices (or (pos, photons) pairs).
        Returns the dict; prints a short report unless verbose=False.
        """
        from .optics import image_rate
        m = self._build()
        invs = inversion if isinstance(inversion, (list, tuple)) else [inversion]
        Y = np.asarray(image).flatten() > 0

        def nll_at(srcs):
            lam = np.full(Y.size, m.dark_count)
            for theta, alpha in srcs:
                pat = image_rate(m.lenses, m.sensor, tuple(theta), 1.0,
                                 m.scint.working_distance).flatten()
                lam += float(alpha) * pat / pat.sum()
            return float(np.sum(np.where(
                Y, -np.log(1.0 - np.exp(-lam) + 1e-9), lam)))

        nll_numpy = nll_at([(iv.position, iv.photons) for iv in invs])
        nll_solver = float(invs[0].nll)
        out = {"nll_solver": nll_solver, "nll_numpy": nll_numpy,
               "model_rel_err": abs(nll_solver - nll_numpy)
               / (1.0 + abs(nll_numpy)),
               "nll_truth": None, "beats_truth": None}
        if truth is not None:
            ts = truth if isinstance(truth, (list, tuple)) and not (
                len(truth) == 2 and np.isscalar(truth[1])) else [truth]
            srcs = []
            for t in ts:
                if isinstance(t, Vertex):
                    srcs.append((tuple(t.position), float(t.photons)))
                else:
                    srcs.append((tuple(t[0]), float(t[1])))
            out["nll_truth"] = nll_at(srcs)
            out["beats_truth"] = nll_numpy <= out["nll_truth"] + 1e-6
        if verbose:
            print(f"validate: NLL solver={nll_solver:.4f}  "
                  f"numpy forward={nll_numpy:.4f}  "
                  f"model rel err={out['model_rel_err']:.2e}")
            if out["nll_truth"] is not None:
                verdict = ("recovered point explains the image at least as "
                           "well as the truth" if out["beats_truth"] else
                           "WORSE than the truth — solver stopped short")
                print(f"validate: NLL truth={out['nll_truth']:.4f}  "
                      f"-> {verdict}")
        return out

    # ---- inverse: the exact model, solved by Rekin -----------------------
    def invert(self, image, sources=1, truth=None, budget=60.0,
               keep_model=None, sparse=False, stall=0.0):
        """Recover `sources` vertices from `image` by handing Rekin the EXACT
        likelihood of the forward model (see `model_export`): same optics, same
        normalisation (alpha = detected photons), constants to
        10 significant digits. No grid search, no separate fitting code — the
        solver's answer is the result.

        Returns an `Inversion` (a list, brightest first, for sources > 1).
        `truth` adds `.error`; `keep_model=path` writes the emitted .rn there.
        `sparse=True` uses the fired-pixel encoding (exact for fine-pitch
        sensors with all blobs on-sensor — see `emit_model_sparse`).
        `stall=N` returns as soon as the incumbent has not improved for N
        seconds (0 = run the full budget).
        """
        from .model_export import solve_model
        m = self._build()
        image = np.asarray(image)
        if not np.count_nonzero(image):
            raise ValueError("image has no signal: every pixel is zero")
        found, status, nll = solve_model(m, image, sources=sources,
                                         budget=budget, keep_model=keep_model,
                                         sparse=sparse, stall=stall)
        found.sort(key=lambda s: -s[1])

        def pos_of(v):
            if v is None:
                return None
            return tuple(v.position) if isinstance(v, Vertex) else tuple(v)

        truths = ([pos_of(truth)] if sources == 1 and truth is not None else
                  [pos_of(v) for v in (truth or [])])
        out = []
        for pos, alpha in found:
            match = None
            if truths:
                d = [float(np.linalg.norm(np.subtract(pos, tt)))
                     for tt in truths]
                match = truths.pop(int(np.argmin(d)))
            out.append(Inversion(pos, alpha, nll, truth=match, status=status))
        return out[0] if sources == 1 else out

    def __repr__(self):
        return (f"Imager(scint={self.scint}, lenses={self.lenses}, "
                f"sensor={self.sensor}, vertices={len(self.vertices)})")


def imager():
    """An empty system. Add a scintillator, a lens array and a detector."""
    return Imager()
