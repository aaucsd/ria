"""Emit the inversion as an exact algebraic model and solve it with Rekin.

COMPACT ENCODING (2026-08-06, default): decision variables are exactly the
physical unknowns (x_j, y_j, z_j, alpha_j per source, alpha in [1, 300]);
the forward chain is written as parse-time macros; the dark side of the
Bernoulli NLL is summed in closed form and the grid normaliser S_j is
factored into per-lens row x column sums, so only FIRED pixels appear
individually.  Exact at every binning — the ANF the solver builds scales
with the fired count, not the pixel count.  (The earlier dense per-pixel
encoding spent ~120 s in parse+ANF on a 4 MB two-source 24x24 model; the
compact file is ~20x smaller for the same likelihood.)

SPARSE ENCODING (invert(sparse=True), fine-pitch sensors): additionally
replaces S_j by its analytic fine-pitch limit sum_i a_i^2/u_j^2 (blur
cancels on integration).  Exact when pixels are much smaller than the blur
floor and every blob is on-sensor; at the realistic 6.9 um pitch the grid mass
matches the analytic value to ~1e-13, while at 24x24 it differs by ~8%,
which is why the compact grid-exact form is the default.

alpha_j is the expected DETECTED photon count; its
[1, 300] box is the instrument's operating regime (given instrument data, not
the answer).  Constants are written to 10 significant digits, so forward
and inverse agree to float round-off: there is no model approximation.
"""
import os
import sys
import tempfile

import numpy as np

FMT = "%.10g"


def emit_model(det, image, sources=1, path=None):
    """Write the exact COMPACT .rn for `image` measured by the system `det`
    (the `_Model` adapter of an `Imager`). Returns the file path.

    Exact at every binning, and small.  Two identities do the work:

    1. The Bernoulli NLL is linear in the rate lam on dark pixels, so the
       dark side of the sensor is summed in closed form:
         sum_dark lam_k = sum_j alpha_j (1 - F_j/S_j) + m_dark * eps
       with F_j the pattern mass on the FIRED pixels only.

    2. S_j — the pattern mass over the whole rectangular grid — SEPARATES:
       each lens's blob is an axis-aligned Gaussian, so its grid sum is
       (row sum) x (column sum).  S_j costs nx+ny exp terms per lens
       instead of nx*ny, and equals the numpy forward normaliser
       `pattern_profile(...).sum()` exactly (same factorisation the
       numpy outer-product evaluation uses).

    Only the fired pixels appear individually, so the ANF the solver builds
    scales with the fired count, not the pixel count.  The likelihood is
    IDENTICAL, term for term, to the forward model at any binning — unlike
    `emit_model_sparse`, which additionally replaces S_j by its fine-pitch
    analytic limit and is exact only there.
    """
    lenses, sensor, scint = det.lenses, det.sensor, det.scint
    S_STAR, WD = sensor.imaging_distance, scint.working_distance
    image = np.asarray(image).flatten()
    fired = np.flatnonzero(image)
    ex, ey = sensor.edges()
    cx = 0.5 * (ex[:-1] + ex[1:]); cy = 0.5 * (ey[:-1] + ey[1:])
    wx2 = (ex[1] - ex[0]) ** 2 / 12.0; wy2 = (ey[1] - ey[0]) ** 2 / 12.0
    area = (ex[1] - ex[0]) * (ey[1] - ey[0])
    m = sensor.nx * sensor.ny
    m_dark = m - len(fired)
    nL = lenses.n
    b = scint.bounds
    K = int(sources)
    eps = det.dark_count
    g = lambda v: FMT % v

    L = ["# Exact inversion model (ria.model_export), COMPACT encoding:",
         f"# {K} source(s) x (x,y,z,alpha), {len(fired)} fired of {m} pixels,",
         f"# {nL} lenses.  The dark-pixel NLL is summed in closed form (it",
         "# is linear in lam) and the grid normaliser S_j separates into",
         "# per-lens row x column sums, so only fired pixels appear",
         "# individually.  Identical to the forward simulation at any",
         "# binning; constants at 10 significant digits; Define lines are",
         "# parse-time macros.",
         "Variables:"]
    for j in range(K):
        L += [f"  Real x{j} : [{g(b[0,0])}, {g(b[0,1])}]",
              f"  Real y{j} : [{g(b[1,0])}, {g(b[1,1])}]",
              f"  Real z{j} : [{g(b[2,0])}, {g(b[2,1])}]",
              f"  Real alpha{j} : [1, 300]"]
    L.append("")
    for j in range(K):
        L.append(f"# -- source {j}: object distance, defocus blur, chief-ray "
                 "centres --")
        L.append(f"Define u{j} := (z{j} + {g(WD)})")
        for i in range(nL):
            f = lenses.focal_lengths[i]; a = lenses.apertures[i]
            lx, ly = lenses.positions[i]
            L.append(f"Define s2_{j}_{i} := (({g(a/2)} * sqrt(({g(S_STAR)}"
                     f"*(u{j} - {g(f)})/({g(f)}*u{j}) - 1)^2 + 1e-12) + "
                     f"{g(lenses.blur_floor)}))^2")
            L.append(f"Define vx_{j}_{i} := (s2_{j}_{i} + {g(wx2)})")
            L.append(f"Define vy_{j}_{i} := (s2_{j}_{i} + {g(wy2)})")
            L.append(f"Define cx_{j}_{i} := ({g(lx)} + ({g(S_STAR)}/u{j})"
                     f"*(x{j} - {g(lx)}))")
            L.append(f"Define cy_{j}_{i} := ({g(ly)} + ({g(S_STAR)}/u{j})"
                     f"*(y{j} - {g(ly)}))")
            L.append(f"Define g_{j}_{i} := ({g(a*a)}/({g(2*np.pi)}"
                     f"*sqrt(vx_{j}_{i}*vy_{j}_{i})*u{j}*u{j}))")
        L.append(f"# S{j}: full-grid pattern mass; each lens's grid sum "
                 "separates into (row sum) x (column sum)")
        for i in range(nL):
            rx = " + ".join(f"exp(-({g(c)} - cx_{j}_{i})^2/(2*vx_{j}_{i}))"
                            for c in cx)
            ry = " + ".join(f"exp(-({g(c)} - cy_{j}_{i})^2/(2*vy_{j}_{i}))"
                            for c in cy)
            L.append(f"Define sx_{j}_{i} := ({rx})")
            L.append(f"Define sy_{j}_{i} := ({ry})")
        L.append(f"Define S{j} := ({g(area)}*("
                 + " + ".join(f"g_{j}_{i}*sx_{j}_{i}*sy_{j}_{i}"
                              for i in range(nL)) + "))")
        L.append(f"# pattern of source {j} at each FIRED pixel")
        for t_, k in enumerate(fired):
            kx, ky = cx[k % sensor.nx], cy[k // sensor.nx]
            terms = [f"g_{j}_{i}*exp(-(({g(kx)} - cx_{j}_{i})^2"
                     f"/(2*vx_{j}_{i}) + ({g(ky)} - cy_{j}_{i})^2"
                     f"/(2*vy_{j}_{i})))" for i in range(nL)]
            L.append(f"Define pat_{j}_{t_} := ({g(area)}*("
                     + " + ".join(terms) + "))")
        L.append(f"# F{j}: mass of source {j} on the fired set")
        L.append(f"Define F{j} := ("
                 + " + ".join(f"pat_{j}_{t_}" for t_ in range(len(fired)))
                 + ")")
    L.append("# per-fired-pixel rate: dark floor + superposed source patterns")
    for t_ in range(len(fired)):
        s = " + ".join(f"alpha{j}*pat_{j}_{t_}/S{j}" for j in range(K))
        L.append(f"Define lam_{t_} := ({g(eps)} + {s})")
    L += ["", "Constraints:"]
    if K > 1:
        L.append("  # break the source-label symmetry")
        for j in range(K - 1):
            L.append(f"  z{j+1} - z{j} >= 0 ~ 1e-6")

    dark = " + ".join(f"alpha{j}*(1 - F{j}/S{j})" for j in range(K))
    logs = " ".join(f"- log(1 - exp(-lam_{t_}) + 1e-9)"
                    for t_ in range(len(fired)))
    L += ["", "Minimize:",
          "  # Bernoulli NLL; the dark sum is closed-form:",
          f"  # sum_dark lam = sum_j alpha_j (1 - F_j/S_j) + m_dark*eps,"
          f" m_dark = {m_dark}",
          f"  {dark} + {g(m_dark * eps)} {logs} ~ 1e-3"]

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".rn"); os.close(fd)
    open(path, "w").write("\n".join(L) + "\n")
    return path


def emit_model_sparse(det, image, sources=1, path=None):
    """Write the exact FIRED-PIXELS .rn for `image` — the encoding for
    realistic-scale sensors, where almost every pixel is dark.

    The Bernoulli NLL is LINEAR in lam on dark pixels, so the whole dark side
    collapses in closed form (no approximation):

      sum_dark lam_k = sum_j alpha_j * (1 - F_j/S_j) + m_dark * eps
        where F_j = sum_{k in fired} pat_{j,k}   (mass on the fired pixels)

    and at fine pixel pitch S_j — the full-sensor pattern mass — has the
    closed form  S_j = sum_i a_i^2/u_j^2  (each lens blob is a normalised
    Gaussian: its density integrates to 1, so the blur cancels; pixel
    convolution preserves mass).  Valid when every blob lies on the sensor
    (all-lens visibility — check `Imager.visibility` at plausible sources)
    and pixels are much smaller than the blur floor (6.9 um << 120 um:
    tail-truncation error ~1e-10).  Under those stated conditions the
    objective equals the dense-grid NLL term for term.

    Only the ~alpha fired pixels appear in the file, so the model size is set
    by the PHOTON BUDGET, not the pixel count — a 32-Mpixel sensor emits a
    smaller model than the dense 12x12 one.
    """
    lenses, sensor, scint = det.lenses, det.sensor, det.scint
    S_STAR, WD = sensor.imaging_distance, scint.working_distance
    image = np.asarray(image).flatten()
    fired = np.flatnonzero(image)
    ex, ey = sensor.edges()
    cx = 0.5 * (ex[:-1] + ex[1:]); cy = 0.5 * (ey[:-1] + ey[1:])
    wx2 = (ex[1] - ex[0]) ** 2 / 12.0; wy2 = (ey[1] - ey[0]) ** 2 / 12.0
    area = (ex[1] - ex[0]) * (ey[1] - ey[0])
    m = sensor.nx * sensor.ny
    m_dark = m - len(fired)
    nL = lenses.n
    b = scint.bounds
    K = int(sources)
    eps = det.dark_count
    g = lambda v: FMT % v

    L = ["# Exact inversion model (ria.model_export),",
         f"# SPARSE fired-pixel encoding: {K} source(s) x (x,y,z,alpha),",
         f"# {len(fired)} fired of {m} pixels, {nL} lenses.  The dark-pixel",
         "# NLL is summed in closed form (it is linear in lam), and S_j is",
         "# the analytic full-sensor mass sum_i a_i^2/u_j^2 — see",
         "# emit_model_sparse for the validity conditions.  Constants at 10",
         "# significant digits; Define lines are parse-time macros.",
         "Variables:"]
    for j in range(K):
        L += [f"  Real x{j} : [{g(b[0,0])}, {g(b[0,1])}]",
              f"  Real y{j} : [{g(b[1,0])}, {g(b[1,1])}]",
              f"  Real z{j} : [{g(b[2,0])}, {g(b[2,1])}]",
              f"  Real alpha{j} : [1, 300]"]
    L.append("")
    for j in range(K):
        L.append(f"# ── source {j}: object distance, defocus blur, chief-ray "
                 "centres ──")
        L.append(f"Define u{j} := (z{j} + {g(WD)})")
        for i in range(nL):
            f = lenses.focal_lengths[i]; a = lenses.apertures[i]
            lx, ly = lenses.positions[i]
            L.append(f"Define s2_{j}_{i} := (({g(a/2)} * sqrt(({g(S_STAR)}"
                     f"*(u{j} - {g(f)})/({g(f)}*u{j}) - 1)^2 + 1e-12) + "
                     f"{g(lenses.blur_floor)}))^2")
            L.append(f"Define vx_{j}_{i} := (s2_{j}_{i} + {g(wx2)})")
            L.append(f"Define vy_{j}_{i} := (s2_{j}_{i} + {g(wy2)})")
            L.append(f"Define cx_{j}_{i} := ({g(lx)} + ({g(S_STAR)}/u{j})"
                     f"*(x{j} - {g(lx)}))")
            L.append(f"Define cy_{j}_{i} := ({g(ly)} + ({g(S_STAR)}/u{j})"
                     f"*(y{j} - {g(ly)}))")
            L.append(f"Define g_{j}_{i} := ({g(a*a)}/({g(2*np.pi)}"
                     f"*sqrt(vx_{j}_{i}*vy_{j}_{i})*u{j}*u{j}))")
        L.append(f"# S{j}: full-sensor pattern mass, closed form (blur "
                 "cancels — density integrates to 1 per lens)")
        L.append(f"Define S{j} := ("
                 + " + ".join(f"{g(lenses.apertures[i]**2)}/(u{j}*u{j})"
                              for i in range(nL)) + ")")
        L.append(f"# pattern of source {j} at each FIRED pixel")
        for t, k in enumerate(fired):
            kx, ky = cx[k % sensor.nx], cy[k // sensor.nx]
            terms = [f"g_{j}_{i}*exp(-(({g(kx)} - cx_{j}_{i})^2"
                     f"/(2*vx_{j}_{i}) + ({g(ky)} - cy_{j}_{i})^2"
                     f"/(2*vy_{j}_{i})))" for i in range(nL)]
            L.append(f"Define pat_{j}_{t} := ({g(area)}*("
                     + " + ".join(terms) + "))")
        L.append(f"# F{j}: mass of source {j} on the fired set")
        L.append(f"Define F{j} := ("
                 + " + ".join(f"pat_{j}_{t}" for t in range(len(fired)))
                 + ")")
    L.append("# per-fired-pixel rate: dark floor + superposed source patterns")
    for t in range(len(fired)):
        s = " + ".join(f"alpha{j}*pat_{j}_{t}/S{j}" for j in range(K))
        L.append(f"Define lam_{t} := ({g(eps)} + {s})")
    L += ["", "Constraints:"]
    if K > 1:
        L.append("  # break the source-label symmetry")
        for j in range(K - 1):
            L.append(f"  z{j+1} - z{j} >= 0 ~ 1e-6")

    dark = " + ".join(f"alpha{j}*(1 - F{j}/S{j})" for j in range(K))
    logs = " ".join(f"- log(1 - exp(-lam_{t}) + 1e-9)"
                    for t in range(len(fired)))
    L += ["", "Minimize:",
          "  # Bernoulli NLL; the dark sum is closed-form:",
          f"  # sum_dark lam = sum_j alpha_j (1 - F_j/S_j) + m_dark*eps,"
          f" m_dark = {m_dark}",
          f"  {dark} + {g(m_dark * eps)} {logs} ~ 1e-3"]

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".rn"); os.close(fd)
    open(path, "w").write("\n".join(L) + "\n")
    return path


def solve_model(det, image, sources=1, budget=60.0, keep_model=None,
                sparse=False, stall=0.0):
    """Emit the exact model and solve it with Rekin's canonical global stack
    (`rekin.solve` — the same entry the rekin binary uses,
    compact encoding and all defaults included). Returns (per-source list of
    (position, alpha), status_name, nll).  `stall=N` returns N seconds after
    the last incumbent improvement (0 = run the full budget)."""
    # Large emitted models (K>=2 at fine binning) segfault when a second
    # OpenMP runtime coexists with insu's in the process; single-threaded
    # OMP is the documented safe setting (the manager threads itself).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.setrecursionlimit(1000000)       # deep parse trees of the inline form
    from rekin import solve as solve_file
    emitter = emit_model_sparse if sparse else emit_model
    path = emitter(det, image, sources=sources, path=keep_model)
    try:
        # inversions are tiny-variable / huge-atom models: the ANF-side
        # machinery contributes nothing here and its CONSTRUCTION alone
        # burns budget — skip it entirely (race = ipm + sa on the tapes).
        # (Building the model via the rekin.Problem API instead of emitting
        # and reparsing text was measured to give NO speedup: after the
        # anf=False + DAG-macro-parse changes the parse is already cheap,
        # and the time is dominated by the shared gradient-tape prep + race.)
        r = solve_file(path, budget=budget, anf=False, stall_stop=stall)
    finally:
        if keep_model is None:
            os.unlink(path)
    sol = r.get("sol")
    if not sol:
        raise RuntimeError(
            f"solver returned no solution (status {r.get('status_name')}); "
            f"raw result keys: {sorted(r.keys())}")
    out = [((sol[f"x{j}"], sol[f"y{j}"], sol[f"z{j}"]), sol[f"alpha{j}"])
           for j in range(sources)]
    return out, r["status_name"], r["obj"]
