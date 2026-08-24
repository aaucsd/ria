"""Example 3 — two sources on a 24x24 sensor.

A gamma that Compton-scatters deposits energy twice; both flashes land on
one sensor and their light superposes (each pixel's rate is the sum of the
two sources' rates).  A single source is already located well at 24x24 (see example 1's
discussion), but TWO sources are not: one is recovered well, while the
other is replaced by a nearer, brighter impostor — multi-source needs
more resolution than one source does.  Example 4 runs the
same two sources on the real camera and locates both.

Run:  python 3_two_sources_coarse.py
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
                     "models", "3_two_sources_coarse.rn")


def title(head, sub=""):
    print("=" * 78); print(head)
    if sub:
        print(textwrap.fill(" ".join(sub.split()), 78))
    print("=" * 78)


def say(text):
    for para in text.strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), 78)); print()


title("Two sources, 24x24 sensor — one recovered, one lost to an impostor")

# ── the instrument (identical to examples 1, 2, 4, 5) ─────────────────────
FS = (6, 7.5, 9, 10.5, 12, 13.5, 15)
LENSES = [{"position": tuple(p), "f": FS[i % 7], "aperture": 1.5}
          for i, p in enumerate(ria.hex_layout(rings=3, pitch=3.6))]

system = ria.imager()
system.addScint(size=(50, 50, 50), working_distance=5.0)
system.addLens(lenses=LENSES, blur_floor=0.12)
show_lenses(LENSES)
system.addDetector(size=(46.15, 32.84), pixels=(24, 24),
                   imaging_distance=20.477, dark_count=1e-9)

# ── the measurement: two vertices superposed in one image ─────────────────
# Passing a list of vertices sums their rates: lambda_i = sum_k alpha_k
# P_i(theta_k).  The detector records no marker of which photon
# came from which vertex — separating them is entirely the inverse problem.
SOURCES = [((5.0, 3.0, 30.0), 200),      # brighter, deeper
           ((-6.0, -2.0, 24.0), 150)]    # dimmer, shallower
verts = [system.addVertex(list(p), photons=n) for p, n in SOURCES]
image = system.forward(verts, seed=3)
print(f"\nmeasurement: {int(image.sum())} of {image.size} one-bit pixels fired "
      "(both sources, 24x24)")

# ── the inversion: recover BOTH vertices jointly ──────────────────────────
# sources=2 emits one block of unknowns (x, y, z, alpha) per source plus a
# depth-ordering row so that swapping the labels is not a second optimum.
# The likelihood splits the light; nothing is placed greedily.
t0 = time.time()
found = system.invert(image, sources=2, truth=verts, budget=120, stall=5,
                      keep_model=MODEL)
print(f"\nsolve time: {time.time() - t0:.0f} s (budget 120 s)")

print("\nresult (brightest first)")
for j, inv in enumerate(found):
    print(f"  source {j+1}: recovered "
          f"({inv.position[0]:6.2f}, {inv.position[1]:6.2f}, {inv.position[2]:6.2f}) mm"
          f"   truth {inv.truth}   error {np.linalg.norm(inv.error):5.2f} mm"
          f"   alpha={inv.photons:.0f}   {inv.status}")

# ── why the second source is displaced ────────────────────────────────────
v = system.validate(found, image, truth=[(p, float(n)) for p, n in SOURCES])
print(f"""
cross-check (independent numpy forward model)
  NLL of recovered pair    {v['nll_solver']:.3f}
  NLL of the TRUE sources   {v['nll_truth']:.3f}
  model vs forward optics   rel. err {v['model_rel_err']:.1e}
""")

say(f"""
The recovered pair explains the image better than the truth does
(beats_truth = {v['beats_truth']}), so once again this is the true maximum
of the likelihood, not a solver miss.  What two sources add over example 1
is a cheaper escape: with alpha free, the solver can park one source nearby
and drive its brightness to the [1, 300] cap.  On a saturating one-bit
pixel that costs almost nothing, so a near, capped-bright source reproduces
the fired pixels of the true far source.  The displaced source shows
alpha = 300, sitting on that bound.
""")

say("""
Note that positions are recovered far more tightly than the per-source
photon split: the geometry of the two blob patterns is firm while the
division of brightness between the sources is soft.  The fix is finer binning — example 4 runs these same two sources at the
realistic 6.9 um pitch and recovers both.
""")
