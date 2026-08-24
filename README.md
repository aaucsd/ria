# RIA — Radiation Image Analyzer

## Installation

Install Rekin first:

```bash
pip install rekin
```

Then install missing dependencies if any (numpy, matplotlib, etc.):

```bash
pip install .       # from this directory
```

Then try any example in `examples/`:

```bash
cd examples && python 0_quick_start.py
```

## Functionalities

A scintillator turns a particle interaction into a flash of light. A lens array
in front of a single-photon sensor turns that flash into a pattern of blobs,
and where the blobs land encodes where the interaction happened.

RIA locates a radiation interaction from a plenoptic image and helps design
the lens array that makes it possible.

| you have | you want | use |
|---|---|---|
| a sensor image | the interaction vertex `(x, y, z)` and its photon count | `imager().invert()` |
| a lens array | how precisely it could locate things | `design_score()`, `crb_map()` |
| a design sketch | the best lens array under manufacture constraints | `optimise_focal_lengths_rekin()` |

---

Layout:

```
ria/
  src/ria/                        the library
  examples/                       runnable examples (below)
  manual.html                     the documentation site (pages in manual/)
  pyproject.toml
```

---

## Quick start

Let's start with the simplest example with just one lens to show the pipeline.
This is `examples/0_quick_start.py`.

```python
import ria

system = ria.imager()
system.addScint(size=(30, 30, 40),      # scintillator volume (x, y, z), mm:
                                        #   x, y centred on the axis (±15 here);
                                        #   z runs from the front face inward
                working_distance=5.0)   # lens plane → front face, mm

system.addLens(lenses=[{"position": (0.0, 0.0),   # lens centre on lens plane, mm
                        "f": 9.0,                 # focal length, mm
                        "aperture": 1.5}],        # lens RADIUS, mm
               blur_floor=0.12)         # residual blur every blob carries, mm

system.addDetector(size=(46.15, 32.84), # sensor active area, mm
                   pixels=(24, 24),     # one-bit pixels across that area
                   imaging_distance=20.477,   # lens plane → sensor, mm
                   dark_count=1e-9)     # background firing rate per pixel

s1 = system.addVertex([2, 2, 20], photons=80)   # a flash: position mm, detected photons
print(system.visibility(s1.position))   # 1 — how many lenses image this point

image = system.forward(s1, seed=1)      # the binary sensor image
print(int(image.sum()))                 # 6 — pixels that fired (1-bit each)

result = system.invert(image, truth=s1, budget=30, stall=5)
print(result)  # Inversion(position=(1.325, 1.940, 17.493)mm, ..., error=2.597mm, status=local)
```

Reading those numbers:

`visibility` counts the lenses that land this point's blob on the sensor —
with one lens it is 1, and 0 would mean the point cannot be measured at any
brightness (`forward()` then raises rather than return a blank image).
80 detected photons light only 6 of the 576 one-bit pixels, because a pixel
saturates at "1" no matter how many photons arrive. And the inversion comes
back 2.6 mm off, almost all of it in depth: a single lens sees depth only
through its own blur, which is exactly why the real instrument is an array.

The same code with 37 lenses (positions on a hexagonal layout, focal lengths
cycling over seven values so different lenses are sharp at different depths)
is example `1_single_source_coarse.py` — still information-limited at its
coarse 12×12 binning, until example `2_single_source_realistic_scale.py` runs the
identical source at a realistic 6.9 µm pitch and recovers it to ~0.05 mm.
Every later example only changes this same handful of calls.

Several vertices in one image use the same syntax; the light superposes, and
you ask for that many sources back (a list, brightest first):

```python
s1 = system.addVertex([5, 3, 30], photons=200)
s2 = system.addVertex([-6, -2, 24], photons=150)
image = system.forward([s1, s2], seed=3)
found = system.invert(image, sources=2, truth=[s1, s2], budget=120, stall=5)
```

## What you get back

`invert()` returns an `Inversion` (a list of them, brightest first, for
`sources > 1`):

| field | meaning |
|---|---|
| `.position` | reconstructed vertex `(x, y, z)`, mm |
| `.photons` | fitted detected-photon count (the model's alpha) |
| `.status` | `delta-global` (proved optimal), `local` (certified local optimum), or `best-effort` |
| `.error` | per-axis absolute error, when a `truth` was supplied |

The status is the solver's own claim about the fit, and it is never rounded
up: `delta-global` means global optimality was proved, `local` means a
certified local optimum, `best-effort` means the best point found in the
budget. In the shipped examples the imaging inversions return `local` (or
`best-effort` on the three-source frontier of example 5); the design solves
of example 6 return `local` and `best-effort`.

To check a result independently of the solver, `validate()` recomputes the
likelihood through the numpy forward optics:

```python
v = system.validate(result, image, truth=[((2, 2, 20), 80.0)])
v["model_rel_err"]                # ~1e-5: the solved model matches the forward optics
v["nll_solver"], v["nll_truth"]   # NLL of the fit and of the truth
v["beats_truth"]                  # the fit explains the image at least as well as the truth
```

### Several vertices in one image

A Compton scatter deposits energy twice, so both flashes land on the same
sensor. The model then contains one block of unknowns per source plus a
depth-ordering constraint so that swapping source labels does not count as a
second optimum. The likelihood decides how to split the light; nothing is
fitted greedily source-by-source.

Separation is reliable when the vertices are well apart, both seen by a good
share of the array, and comparably bright. It degrades when one source is
much dimmer or sits in a poorly-imaged corner — the brighter fit absorbs its
light. Check `visibility()` per vertex before trusting a separation, and note
that positions come back much tighter than per-source photon counts: on a
saturating one-bit sensor the geometry of the blob patterns is firm while the
brightness split between sources is soft.

---

## Designing the array

The other half of the instrument. Depth precision has a floor — the
Cramér–Rao bound — set by the optics alone, before any algorithm:

```python
array = ria.LensArray(lenses=LENSES, blur_floor=0.12,
                      imaging_distance=20.477, sensor_size=(46.15, 32.84))

radii  = (5, 10, 15, 20, 25)        # test grid: lateral radius, mm
depths = tuple(range(10, 51, 5))    # and depth, mm
ria.design_score(array, radii, depths, photons=100)  # mean log CRB (lower = better)
_, _, M = ria.crb_map(array, radii, depths, 100)     # the per-point map behind the figures
```

Each lens's focal length is a vote for one depth: it is sharp only where its
image distance matches the sensor distance. A lens contributes most depth
information when it is sharp and far off-axis (a longer lever arm).

The design problem — choose focal lengths inside the manufacturable
`[3, 16] mm` — is posed to Rekin in two forms (example 6):

```python
best, status, score = ria.optimise_focal_lengths_rekin(
    array, radii, depths, photons, f_range=(3.0, 16.0), budget=240)   # mean log CRB
best, status, t = ria.optimise_focal_lengths_rekin(
    array, radii, depths, photons, f_range=(3.0, 16.0), budget=180,
    objective="minimax")                                  # worst-case log CRB
```

The mean objective matches a coordinate-descent sweep
(`ria.optimise_focal_lengths`) on its own terms. The minimax objective —
minimise the worst log CRB anywhere in the volume — is a query the sweep
cannot even pose: the shipped example improves worst-case depth precision
2.3× over the starting array (returned `best-effort` within its budget) and
cross-checks every number against an independent numpy Fisher
implementation.

Two caveats about this CRB. The CRB here is the
depth-only Fisher information (the z–z entry, with x, y treated as known),
so it is slightly optimistic where depth and lateral position are
correlated. And the CRB assumes Poisson photon counting while the actual
readout is one-bit binary, so CRB figures are a floor for a counting sensor,
not a prediction for the binary one — the two should not be compared
directly.

---

## Examples

Each example in `examples/` is a standalone script — run it with
`python <name>.py` from that directory. Every one prints a narrated report
as it goes (including the solve time), and the inversion examples write the
exact `.rn` model they solved into `examples/models/`, so you can inspect
the algebra or re-solve it with `rekin <model>.rn`. Examples 1 through 5
share one instrument — a 37-lens hexagonal array — and change only the
sensor binning and the number of sources, so together they trace what
plenoptic localisation can and cannot do.

`0_quick_start.py` — one lens, one vertex, the whole pipeline. A single lens
sees depth only through one blob's blur, so the reconstruction is ~2.6 mm
off, almost all in depth; this is what the array in the later examples fixes.

`1_single_source_coarse.py` — one source on a coarse 12x12 sensor. The
reconstruction lands ~20 mm off, and `validate()` shows why it is not a
solver failure: the recovered point explains the image at least as well as
the truth does. A 144-pixel one-bit image is genuinely ambiguous.

`2_single_source_realistic_scale.py` — the same source on a realistic sensor's
6688 x 4760 pixels at 6.9 µm pitch. With one photon per pixel the sensor no
longer saturates and the source is recovered to ~0.05 mm. Only the fired
pixels enter the model (sparse encoding), so its size scales with the photon
budget, not the pixel count.

`3_two_sources_coarse.py` — two sources on a 24x24 sensor. A single source
is already located well at 24x24, but two are not: one is recovered well,
the other is replaced by a nearer, brighter impostor that rides the alpha
cap — a genuine likelihood maximum, confirmed by `validate()`, not a solver
miss. Multi-source needs more resolution than one source does.

`4_two_sources_realistic_scale.py` — the same two sources at realistic-scale pitch.
Both are located to a millimetre or better and the fitted photon counts come
back near the truth: with no saturation the impostor has no purchase.

`5_three_sources_realistic_scale.py` — three sources at realistic-scale pitch: the
current frontier. The physics is favourable (example 4 shows two separate
cleanly), but the twelve-variable joint search is strongly multimodal and
returns a best-effort incumbent within the budget rather than a certified
optimum. The example reports this honestly. This is an optimisation limit,
not an information one: more pixels do not help (already at realistic scale) and a
wider lens array only halves the error without closing the search.

`6_lens_design.py` — design the instrument. Choose the focal lengths to make
depth as measurable as possible everywhere: first the mean log CRB (checked
against an independent coordinate sweep, which it matches), then the minimax
design — a query the sweep cannot pose — which improves the worst-case
precision 2.3x over the starting array within its solve budget.

The documentation site is [`manual.html`](manual.html); its reference pages
live in `manual/`:

- [Building a system](manual/system-api.html) — every call, every parameter
- [Lens model](manual/lens-model.html) — how lens properties are defined
- [The sensor image](manual/sensor-image.html) — the array format and how it is formed
- [Image → solver](manual/solver-interface.html) — how the image is fitted
- [The constraint file](manual/constraint-file.html) — the full algebraic model

---

## Forward and backward

There is one forward model, and the inversion uses it exactly.

Forward (`forward`): a source at `(x, y, z)` delivering `alpha` detected
photons produces an expected count per pixel

```
lam(u,v) = eps + alpha * P(u,v; x,y,z)          with  sum_uv P = 1
```

where `P` comes from the chief-ray optics: each lens paints a Gaussian blob
(centre from the magnification `S*/u`, width from defocus plus the pixel's
own second moment `w^2/12`), evaluated as a density at the pixel centre
times the pixel area, normalised over the sensor. Every operation is
`exp / sqrt / ratio` arithmetic. Each pixel then fires with probability
`1 - e^-lam` — one bit, sampled with an explicit seed.

Inverse (`invert`): the same expressions — same blob formula, same
normalisation, constants written to 10 significant digits — are emitted as
an algebraic model whose objective is the Bernoulli likelihood:

```
NLL = sum_dark lam(u,v) - sum_fired log(1 - e^-lam + 1e-9)
```

and handed to the installed `rekin` solver. There is no grid search, no
separate fitter, and no approximation gap: the model the solver optimises is
the model that generated (or measured) the data, term for term. This is why
the forward model uses a pixel-convolved Gaussian density rather than an
error-function integral — the erf is not in the solver's operator set, and
an inverse that only approximates the forward model answers a slightly
different question. `keep_model="event.rn"` on `invert()` writes the exact
emitted model out for inspection; every shipped example does this, so after
a run you will find the very `.rn` file Rekin solved in `examples/models/`.

For fine-pitch sensors, `invert(sparse=True)` uses the fired-pixel encoding:
the Bernoulli likelihood is linear in the rate on dark pixels, so the dark
side of the sensor is summed in closed form and only fired pixels appear in
the model. The model size then scales with the photon budget rather than the
pixel count. This is exact when pixels are much smaller than the blur floor
and every blob lies on the sensor; at coarse binning the default COMPACT
encoding is used instead (dark pixels still summed in closed form, the grid
normaliser written as per-lens row x column sums).

Because the pattern is normalised, the fitted `alpha` is the expected
detected photon count — directly comparable to what you simulated with. The
number of fired pixels is not the photon count: a one-bit pixel saturates,
so `sum(image)` under-counts (77 photons → 14 fired at 12 x 12, 42 at
24 x 24; only at the realistic 6.9 µm pitch does one photon get one pixel).
The model bounds alpha to `[1, 300]`, the instrument's operating regime.

## Measured accuracy

Same 37-lens instrument, cold start over the full volume, as run by the
shipped examples:

| example | sources | binning | 3-D error |
|---|--:|---|---|
| 1 | 1 | 12 x 12 | ~20 mm (information-limited) |
| 2 | 1 | realistic 6.9 µm | ~0.05 mm |
| 3 | 2 | 24 x 24 | one ~3 mm, one lost to an impostor |
| 4 | 2 | realistic 6.9 µm | both ~0.1–1.5 mm |
| 5 | 3 | realistic 6.9 µm | best effort |

The coarse-binning error (examples 1, 3) is a property of the sensor, not
the solver: `validate()` shows the wrong reconstruction explains the coarse
image at least as well as the truth does. One source is already located well
at 24 x 24, but two are not — multi-source needs more resolution. At
realistic-scale pitch the one-bit sensor stops saturating and one and two sources
are located precisely (examples 2, 4). Three sources (example 5) is limited
not by the sensor but by the twelve-variable global search, which returns a
best-effort incumbent within the budget. Depth is always the weakest axis, as
the Cramér–Rao analysis predicts.

## Model fidelity

The optics are the paraxial chief-ray approximation: exact on-axis, but a
real lens draws an ellipse off-axis where this model draws a circle, and it
over-trusts shallow-depth focus. For design studies that is the right trade
(closed form, millions of evaluations); for final numbers, calibrate against
measurement.


## For LLM agents

If you are an agent building on this package — writing new benchmark
instances, emitting `.rn` models in the same style, or driving the solver
directly — read [`agent-instructions/`](agent-instructions/) first. It
specifies the solver's exact file format and API, the likelihood encoding
macro by macro, and the rules for generating sound benchmarks. It is written
for machines; the human documentation is `manual.html`.