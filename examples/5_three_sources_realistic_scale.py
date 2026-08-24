"""Example 5 — three sources at the camera's realistic-scale pitch: the current frontier.

Example 4 showed that at realistic-scale pitch the PHYSICS of multi-source separation
is favourable: with no sensor saturation, two sources are located to about a
millimetre.  Three sources is where the SOLVER, not the physics, becomes the
limit.  The inversion is now a global search over twelve unknowns (x, y, z,
alpha for each of three sources), and the likelihood is strongly multimodal.
Within a practical budget the solver returns a best-effort incumbent rather
than a certified optimum: some sources are recovered, others are displaced,
and the incumbent may not even beat the true configuration's likelihood.

This example runs that case honestly and reports exactly what comes back.
It is not tuned to succeed; it shows where the method currently stops.

Run:  python 5_three_sources_realistic_scale.py   (several minutes)
"""
import os
import textwrap
import time

import numpy as np
import ria


def show_lenses(lenses):
    """Print the lens array as a readable table — the {position, f, aperture}
    records are the exact configuration the forward model and solver use."""
    fs = sorted({L["f"] for L in lenses})
    aps = sorted({L["aperture"] for L in lenses})
    print(f"\nlens array: {len(lenses)} lens(es), aperture radius "
          f"{aps[0] if len(aps) == 1 else aps} mm, focal lengths {fs} mm")
    print("  each record = {position: (x, y) mm, f: focal length mm, "
          "aperture: radius mm}")
    print(f"  {'idx':>3}  {'x(mm)':>7}  {'y(mm)':>7}  {'f(mm)':>6}  {'aperture':>8}")
    for i, L in enumerate(lenses):
        x, y = L["position"]
        print(f"  {i:>3}  {x:7.2f}  {y:7.2f}  {L['f']:6.1f}  {L['aperture']:8.1f}")


MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", "5_three_sources_realistic_scale.rn")
BUDGET = 300.0        # solver time budget, seconds (a deliberate timeout)


def title(head, sub=""):
    print("=" * 78); print(head)
    if sub:
        print(textwrap.fill(" ".join(sub.split()), 78))
    print("=" * 78)


def say(text):
    for para in text.strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), 78)); print()


title("Three sources at realistic-scale pitch — a best-effort frontier")

# ── the instrument (identical to examples 1-4) ────────────────────────────
FS = (6, 7.5, 9, 10.5, 12, 13.5, 15)
LENSES = [{"position": tuple(p), "f": FS[i % 7], "aperture": 1.5}
          for i, p in enumerate(ria.hex_layout(rings=3, pitch=3.6))]

system = ria.imager()
system.addScint(size=(50, 50, 50), working_distance=5.0)
system.addLens(lenses=LENSES, blur_floor=0.12)
show_lenses(LENSES)
system.addDetector(size=(46.15, 32.84), pixels=(6688, 4760),
                   imaging_distance=20.477, dark_count=1e-9)

# ── the measurement: three vertices superposed ────────────────────────────
SOURCES = [((6.0, 4.0, 20.0), 150),
           ((-8.0, 6.0, 28.0), 150),
           ((-2.0, -9.0, 34.0), 150)]
verts = [system.addVertex(list(p), photons=n) for p, n in SOURCES]
image = system.forward(verts, seed=11)
print(f"\nmeasurement: {int(image.sum())} of {image.size} pixels fired "
      "(three sources, 6.9 um pitch)")

# ── the inversion: twelve unknowns, a hard global search ──────────────────
t0 = time.time()
found = system.invert(image, sources=3, truth=verts, budget=BUDGET, stall=15,
                      sparse=True, keep_model=MODEL)
solve_seconds = time.time() - t0

print(f"\nsolve time: {solve_seconds:.0f} s (budget {BUDGET:.0f} s)")
print("result (brightest first)")
for j, inv in enumerate(found):
    print(f"  source {j+1}: recovered "
          f"({inv.position[0]:6.2f}, {inv.position[1]:6.2f}, {inv.position[2]:6.2f}) mm"
          f"   truth {inv.truth}   error {np.linalg.norm(inv.error):5.2f} mm"
          f"   alpha={inv.photons:.0f}   {inv.status}")

# ── read the result honestly ──────────────────────────────────────────────
v = system.validate(found, image, truth=[(p, float(n)) for p, n in SOURCES])
print(f"""
cross-check (independent numpy forward model)
  NLL of recovered set     {v['nll_solver']:.1f}
  NLL of the TRUE sources   {v['nll_truth']:.1f}
  recovered beats truth?    {v['beats_truth']}
""")

verdict = ("the incumbent explains the image at least as well as the truth"
           if v["beats_truth"] else
           "the incumbent does NOT yet beat the truth — the search stopped short")
say(f"""
Status {found[0].status}: {verdict}.  Unlike examples 1 and 3, where the
recovered point always beat the truth (an information limit), the failure
here is one of optimisation.  Twelve coupled unknowns give the likelihood
many comparable basins, and simulated annealing plus interior-point polish
does not reliably reach the global one inside {BUDGET:.0f} seconds; the
result is returned as best-effort, without a local-optimum certificate.
""")

say("""
Why more of the obvious knobs do not fix it: the sensor is already at realistic-scale
6.9 um pitch, so there is no saturation left to remove and finer pixels add
nothing (example 4 confirms two sources separate cleanly here).  A wider,
denser lens array sharpens the likelihood and roughly halves the errors, but
still returns best-effort — it improves conditioning without closing the
search.  Three-source joint reconstruction is the honest frontier of this
tool: the physics permits it, and reaching it reliably is a question of
better global optimisation, not more photons or pixels.
""")
