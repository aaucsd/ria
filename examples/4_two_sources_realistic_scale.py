"""Example 4 — the same two sources as example 3, on the real camera.

The camera is 6688 x 4760 pixels at 6.9 um pitch [P Sec. V].  At that
resolution each photon lands on its own pixel, so the one-bit sensor no
longer saturates, and the near/bright impostor that captured the second
source at 12x12 (example 3) is gone: both vertices are located to about a
millimetre or better, and the fitted photon counts come back close to the
true detected counts.

The 32-million-pixel image is handled by the fired-pixel encoding, the same
one example 2 uses: only the few hundred pixels that actually fire enter the
model, and the dark side of the likelihood is summed in closed form, so the
emitted model is small.

Run:  python 4_two_sources_realistic_scale.py   (about three minutes)
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
                     "models", "4_two_sources_realistic_scale.rn")


def title(head, sub=""):
    print("=" * 78); print(head)
    if sub:
        print(textwrap.fill(" ".join(sub.split()), 78))
    print("=" * 78)


def say(text):
    for para in text.strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), 78)); print()


title("Two sources at the camera's realistic 6.9 um pitch — both located")

# ── the instrument (identical to examples 1-3, 5) ─────────────────────────
FS = (6, 7.5, 9, 10.5, 12, 13.5, 15)
LENSES = [{"position": tuple(p), "f": FS[i % 7], "aperture": 1.5}
          for i, p in enumerate(ria.hex_layout(rings=3, pitch=3.6))]

system = ria.imager()
system.addScint(size=(50, 50, 50), working_distance=5.0)
system.addLens(lenses=LENSES, blur_floor=0.12)
show_lenses(LENSES)
# 6688 x 4760 pixels over 46.15 x 32.84 mm = 6.9 um pitch [P Sec. V].
system.addDetector(size=(46.15, 32.84), pixels=(6688, 4760),
                   imaging_distance=20.477, dark_count=1e-9)

# ── the measurement: the SAME two sources as example 3 ────────────────────
SOURCES = [((5.0, 3.0, 30.0), 200),
           ((-6.0, -2.0, 24.0), 150)]
verts = [system.addVertex(list(p), photons=n) for p, n in SOURCES]
image = system.forward(verts, seed=3)
print(f"\nmeasurement: {int(image.sum())} of {image.size} pixels fired "
      "(both sources, 6.9 um pitch)")

# ── the inversion: sparse=True selects the fired-pixel encoding ───────────
# Everything else is unchanged: cold start over the whole volume, alpha in
# [1, 300], no initial guess.
t0 = time.time()
# NO stall here: this model's search plateaus in a decoy basin (~7 mm)
# for ~70 s before it jumps to the true basin at ttb~115 s, so a short
# stall would stop early on the wrong answer.  It runs the full budget.
found = system.invert(image, sources=2, truth=verts, budget=180,
                      sparse=True, keep_model=MODEL)
print(f"\nsolve time: {time.time() - t0:.0f} s (budget 180 s)")

print("\nresult (brightest first)")
for j, inv in enumerate(found):
    print(f"  source {j+1}: recovered "
          f"({inv.position[0]:6.2f}, {inv.position[1]:6.2f}, {inv.position[2]:6.2f}) mm"
          f"   truth {inv.truth}   error {np.linalg.norm(inv.error):5.2f} mm"
          f"   alpha={inv.photons:.0f}   {inv.status}")

v = system.validate(found, image, truth=[(p, float(n)) for p, n in SOURCES])
print(f"""
cross-check (independent numpy forward model)
  NLL of recovered pair    {v['nll_solver']:.3f}
  NLL of the TRUE sources   {v['nll_truth']:.3f}
  model vs forward optics   rel. err {v['model_rel_err']:.1e}
""")

say("""
Both sources are now within about a millimetre, and the fitted photon
counts are close to the true detected counts (200 and 150) rather than
pinned at the cap.  The impostor of example 3 has no purchase here: with one
photon per pixel the sensor no longer saturates, so over-brightening a near
source no longer reproduces the image for free.  The information that was
missing at 12x12 is present at realistic-scale pitch, and the same solver — same cold
start, same volume box — now separates the two vertices.
""")
