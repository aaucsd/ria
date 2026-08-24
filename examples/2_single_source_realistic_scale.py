"""Example 2 — the same single source as example 1, on the real camera.

Example 1 showed that a coarse 12x12 sensor cannot locate a source: the
image is genuinely ambiguous.  Here the same source and the same 77 photons
are measured on the camera's true 6688 x 4760 pixels at 6.9 um pitch
[P Sec. V], and the reconstruction sharpens to about 0.05 mm.

A dense per-pixel model at 32 megapixels is neither possible nor needed:
with 77 photons at most 77 pixels ever fire, and the dark side of the
Bernoulli likelihood is linear in the rate, so the whole dark sensor
collapses into one closed-form term.  Only the fired pixels enter the model
(sparse=True), so its size scales with the photon budget, not the pixel
count.

Run:  python 2_single_source_realistic_scale.py
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
                     "models", "2_single_source_realistic_scale.rn")


def title(head, sub=""):
    print("=" * 78); print(head)
    if sub:
        print(textwrap.fill(" ".join(sub.split()), 78))
    print("=" * 78)


def say(text):
    for para in text.strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), 78)); print()


title("One source at the camera's realistic 6.9 um pitch — sub-0.1 mm localisation")

# ── the instrument (identical to examples 1, 3, 4, 5) ─────────────────────
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

# ── the measurement: the SAME source as example 1 ─────────────────────────
TRUTH, PHOTONS = (2.0, 2.0, 31.0), 77
s1 = system.addVertex(list(TRUTH), photons=PHOTONS)
image = system.forward(s1, seed=1)
print(f"\nmeasurement: {int(image.sum())} of {image.size} pixels fired "
      "(6.9 um pitch)")
say("""
At 6.9 um pitch no two photons share a pixel, so the fired count is about
the photon count and the one-bit sensor no longer saturates — the
information the coarse image lacked is present here.
""")

# ── the inversion: sparse=True selects the fired-pixel encoding ───────────
t0 = time.time()
found = system.invert(image, truth=s1, budget=120, sparse=True, stall=5,
                      keep_model=MODEL)
print(f"solve time: {time.time() - t0:.0f} s (budget 120 s)")
err3 = float(np.linalg.norm(found.error))

print(f"""result
  truth      ({TRUTH[0]:6.2f}, {TRUTH[1]:6.2f}, {TRUTH[2]:6.2f}) mm,  alpha = {PHOTONS}
  recovered  ({found.position[0]:6.2f}, {found.position[1]:6.2f}, {found.position[2]:6.2f}) mm,  alpha = {found.photons:.0f}
  gap        ({found.error[0]:6.3f}, {found.error[1]:6.3f}, {found.error[2]:6.3f}) mm  ->  {err3:.3f} mm in 3-D
  status     {found.status}
  model      {os.path.basename(MODEL)}
""")

say(f"""
The source is recovered to {err3:.3f} mm from a cold start over the whole
50 mm volume — no initial guess, no calibration grid, no separate fitter.
The fitted alpha ({found.photons:.0f}) is close to the true detected count
({PHOTONS}) rather than pinned at a bound, because at realistic-scale pitch the
brightness is identifiable too.  This is the same source that example 1
could not locate at 12x12; only the sensor binning changed.
""")
