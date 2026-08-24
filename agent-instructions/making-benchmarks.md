# Making new benchmarks from RIA (for agents)

Recipes for generating new, *correct* benchmark instances — either through
the RIA API (preferred: the library emits the model for you) or by writing
`.rn` files directly in the encoding of `ria-encoding.md`.

## Route 1 — generate through the API (preferred)

```python
import ria

FS = (6, 7.5, 9, 10.5, 12, 13.5, 15)
system = ria.imager()
system.addScint(size=(50, 50, 50), working_distance=5.0)
system.addLens(lenses=[{"position": tuple(p), "f": FS[i % 7], "aperture": 1.5}
                       for i, p in enumerate(ria.hex_layout(rings=3, pitch=3.6))],
               blur_floor=0.12)
system.addDetector(size=(46.15, 32.84), pixels=(12, 12),
                   imaging_distance=20.477, dark_count=1e-9)

s1 = system.addVertex([2, 2, 31], photons=77)      # ground truth, by construction
image = system.forward(s1, seed=1)                 # deterministic given the seed
found = system.invert(image, truth=s1, budget=90, stall=5,
                      keep_model="bench_001.rn")   # <- the benchmark file
```

`keep_model=` writes the exact model that was solved: that file **is** the
benchmark instance. `seed` makes the measurement — and therefore the emitted
file — fully reproducible.

### Knobs that produce genuinely different instances

| knob | effect on difficulty |
|---|---|
| `pixels=(n, n)` | coarse (12×12, 24×24) → saturation ambiguity, impostor optima; fine (6.9 µm pitch, use `sparse=True`) → well-identified |
| `sources=K` | model has `4K` variables; K=2 is solvable at fine pitch, K=3 is a hard multimodal frontier |
| source geometry | close pairs / dim-vs-bright pairs / low-`visibility()` corners are harder |
| `photons` | fewer fired pixels → flatter likelihood |
| lens count / focal spread | fewer lenses or uniform `f` → weaker depth information |
| `seed` | new measurement noise realisation, same physics |

Before accepting an instance: `system.visibility(pos) > 0` for every source
(a zero-visibility source produces no signal; `forward()` raises).

## Route 2 — write `.rn` directly

Follow `ria-encoding.md` term for term. Constants to 10 significant digits.
Only fired pixels get `pat`/`lam` lines. Keep the macro naming
(`u0, s2_0_0, vx_0_0, cx_0_0, g_0_0, pat_0_p, sx_0_0, sy_0_0, S0, F0,
lam_p`) so files remain diffable against library-emitted ones. Cross-check by
emitting the same configuration through Route 1 and diffing.

## Rules for a sound benchmark

1. **Never encode the answer in a constraint.** The quantity under test must
   be solved for, not pinned. Record the expected value in a comment only:

   ```
   # CHECK x0 = 2.03 +- 0.1   (truth (2,2,31); information-limited at 12x12)
   ```

   A runner solves the file, reads `sol` from the JSON, and compares against
   the CHECK line. A satisfiable status alone proves nothing about values.

2. **State the expected status class, not exact numbers.** `local` vs
   `best-effort` is meaningful; the 10th digit of a `best-effort` incumbent
   is not. Coarse-binning instances may legitimately beat the truth's
   likelihood while being metres off — that is the physics, and
   `validate()`-style independent re-evaluation is the right check, not
   position error alone.

3. **Ground truth by construction.** Generate with `forward(seed=...)` from
   known vertices; store truth positions/photons in comments. Never tune an
   instance until the solver "gets it right" — the interesting instances are
   the ones where the honest answer is `best-effort` or an impostor optimum.

4. **Determinism.** Fix every seed. A benchmark file must re-emit
   byte-identically from its generating script.

5. **Budgets in the runner, not the file.** The `.rn` carries the model;
   pass `--budget`/`--stall` at solve time so the same instance can be run
   at several budgets.

## Difficulty ladder (as established by the shipped examples)

| tier | configuration | expected outcome |
|---|---|---|
| easy | 1 source, fine pitch, `sparse=True` | `local`, ~0.05 mm |
| medium | 1–2 sources, coarse 12–24 px | `local`, beats-truth ambiguity, impostor risk |
| hard | 2 sources, one dim or low-visibility | impostor optimum absorbs the dim source |
| frontier | 3 sources, fine pitch | `best-effort`, incumbent may not beat truth |
