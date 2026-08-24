"""The quick-start example from the README: one lens, one vertex, the whole
pipeline.  A single lens has almost no depth leverage — the point of the
instrument is the array — but every later example uses exactly these calls,
only with more lenses and more sources.
"""
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


# ── the system ───────────────────────────────────────────────────────────
system = ria.imager()
system.addScint(size=(30, 30, 40),      # scintillator volume (x, y, z), mm:
                                        #   x, y centred on the axis (±15 here);
                                        #   z runs from the front face inward
                working_distance=5.0)   # lens plane → front face, mm

LENSES = [{"position": (0.0, 0.0),   # lens centre on lens plane, mm
           "f": 9.0,                 # focal length, mm
           "aperture": 1.5}]         # lens RADIUS, mm
system.addLens(lenses=LENSES,
               blur_floor=0.12)      # residual blur every blob carries, mm
show_lenses(LENSES)

system.addDetector(size=(46.15, 32.84), # sensor active area, mm
                   pixels=(24, 24),     # one-bit pixels across that area
                   imaging_distance=20.477,   # lens plane → sensor, mm
                   dark_count=1e-9)     # background firing rate per pixel

# ── a measurement ────────────────────────────────────────────────────────
s1 = system.addVertex([2, 2, 20], photons=80)   # a flash: position mm, detected photons

vis = system.visibility(s1.position)
print(f"visibility: {vis} of {system.lenses.n} lens(es) image this point")
print("  (0 would mean the point is unmeasurable at any brightness;")
print("   forward() would raise rather than return a blank image)")

image = system.forward(s1, seed=1)      # the binary sensor image
print(f"\nmeasurement: {int(image.sum())} of {image.size} one-bit pixels fired")
print("  80 detected photons light only a handful of pixels because a")
print("  1-bit pixel saturates: it reads 1 whether one photon or ten arrive.")

# ── the inversion ────────────────────────────────────────────────────────
t0 = time.time()
result = system.invert(image, truth=s1, budget=30, stall=5)
print(f"solve time: {time.time() - t0:.0f} s (budget 30 s)\n")

print("\nresult:", result)
print(f"""
Reading the result:
  truth      ({s1.position[0]:7.3f}, {s1.position[1]:7.3f}, {s1.position[2]:7.3f}) mm,  alpha = 80
  recovered  ({result.position[0]:7.3f}, {result.position[1]:7.3f}, {result.position[2]:7.3f}) mm,  alpha = {result.photons:.1f}
  gap        ({result.error[0]:7.3f}, {result.error[1]:7.3f}, {result.error[2]:7.3f}) mm   -> {float(np.linalg.norm(result.error)):.3f} mm in 3-D
  status     {result.status}
""")
print("Why the gap looks like this: the lateral (x, y) error is a fraction")
print("of a millimetre, but nearly all of the 3-D error is in depth z.")
print("One lens sees depth only through how blurred its single blob is,")
print("and a handful of fired pixels pins blur weakly, so along z the")
print("likelihood is nearly flat: this IS the best answer the data")
print("supports, not a solver shortfall.  status=local means Rekin")
print("certified the point as a local optimum of the exact likelihood")
print("(delta-global would mean proved best possible; best-effort would")
print("mean no certificate).  The fitted alpha rides its upper bound for")
print("the same reason: with every fired pixel saturated, brightness")
print("above ~80 changes the image probability almost nothing.")
print()
print("A single lens is the root of the depth error: it sees z only through")
print("one blob's blur.  The real instrument is an ARRAY — many lenses at")
print("different focal lengths, whose blobs shift and sharpen at different")
print("rates with depth.  Examples 1 and 2 use a 37-lens array and study")
print("what still limits accuracy once the array is in place: the sensor.")
