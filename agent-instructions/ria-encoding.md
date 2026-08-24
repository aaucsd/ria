# RIA → `.rn` encoding (for agents)

How `invert()` writes the imaging likelihood as a Rekin model. Read this next
to a real emitted file (`../examples/models/*.rn`) — every name below appears
there. `_j` indexes sources, `_i` lenses; the emitted files write them as
`pat_0_5` = source 0, fired pixel 5, etc.

## The problem being encoded

A source at `(x, y, z)` (mm, inside the scintillator) delivering `alpha`
detected photons produces an expected count per pixel

```
lam(u,v) = eps + alpha * P(u,v; x,y,z)       with  sum_uv P = 1
```

where `P` is a sum of per-lens Gaussian blobs. Each pixel of the one-bit
sensor fires with probability `1 − e^−lam`. Given the observed binary image,
the model minimises the Bernoulli negative log-likelihood over the source
parameters. The same expressions produce the forward simulation — there is no
approximation between the two.

## Decision variables (per source j)

```
Real xj : [x_min, x_max]      # the scintillator box from addScint
Real yj : [y_min, y_max]
Real zj : [1e-06, z_max]
Real alphaj : [1, 300]        # detected photons; the instrument's operating regime
```

Nothing else is a variable. Every derived quantity is a `Define` macro, so
the solver sees one closed-form objective over `4*K` unknowns, cold-started
on the whole box.

## The macro chain (COMPACT encoding — the default)

In emission order, for source `j` and lens `i`:

```
Define uj := (zj + WD)                 # object distance; WD = working_distance
Define s2_j_i := ((a_i/2 * sqrt((S*·(uj−f_i)/(f_i·uj) − 1)^2 + 1e-12) + sigma0))^2
                                       # defocus blur (smoothed |.|), + floor, squared
Define vx_j_i := (s2_j_i + wx^2/12)    # per-axis variance; pixel second moment folded in
Define vy_j_i := (s2_j_i + wy^2/12)
Define cx_j_i := (Lx_i + (S*/uj)*(xj − Lx_i))   # chief-ray blob centre
Define cy_j_i := (Ly_i + (S*/uj)*(yj − Ly_i))
Define g_j_i  := (a_i^2/(2π·sqrt(vx_j_i·vy_j_i)·uj^2))   # amplitude (solid angle / norm)

Define pat_j_p := (A_px · Σ_i g_j_i·exp(−((ux_p−cx_j_i)^2/(2vx_j_i)
                                        + (uy_p−cy_j_i)^2/(2vy_j_i))))
                                       # one per FIRED pixel p; A_px = pixel area,
                                       # (ux_p, uy_p) = that pixel's centre (constants)

Define sx_j_i := (Σ_columns exp(−(ux−cx_j_i)^2/(2vx_j_i)))   # per-lens row/column sums:
Define sy_j_i := (Σ_rows    exp(−(uy−cy_j_i)^2/(2vy_j_i)))   # the grid mass separates
Define Sj := (A_px · Σ_i g_j_i·sx_j_i·sy_j_i)                # full-grid pattern mass
Define Fj := (Σ_fired pat_j_p)                               # fired-pixel mass

Define lam_p := (eps + Σ_j alphaj·pat_j_p/Sj)                # rate at fired pixel p
```

Only **fired** pixels get `pat`/`lam` lines. All constants are written to 10
significant digits, so the emitted model matches the library's numpy forward
computation to float round-off (`validate()` checks exactly this).

## Constraints

- One source: **none** (the Constraints header is present but empty — `Sj`
  is a macro, nothing needs enforcing).
- K sources: one ordering row per adjacent pair, killing label symmetry:

```
Constraints:
  z1 - z0 >= 0 ~ 1e-6
```

## Objective

The dark-pixel part of the NLL is linear in `lam`, so it is summed in closed
form; only fired pixels contribute log terms:

```
Minimize:
  Σ_j alphaj·(1 − Fj/Sj) + m_dark·eps  −  Σ_fired log(1 − exp(−lam_p) + 1e-9)  ~ 1e-3
```

`m_dark` = number of dark pixels (a constant). The `+ 1e-9` guards the log.

## SPARSE encoding (`invert(sparse=True)`)

One change only: at fine pixel pitch the full-grid mass has a closed form —
each blob is a normalised Gaussian, the blur integrates out — so

```
Define Sj := (Σ_i a_i^2/uj^2)
```

replaces the row×column grid sums. Valid when the pixel is much smaller than
the blur floor and every blob lies on the sensor; then the file scales with
the photon budget, not the pixel count (a 32-Mpixel sensor emits a ~400-line
file). At coarse binnings the grid aliases the mass, so COMPACT (exact at any
binning) is the default.

## Tolerances used

| line | `~ tol` |
|---|---|
| source-ordering constraint | `1e-6` |
| objective | `1e-3` |

## Reference instrument values (used by the examples)

| symbol | value | argument |
|---|---|---|
| S\* (lens plane → sensor) | 20.477 mm | `imaging_distance` |
| sensor active area | 46.15 × 32.84 mm | `size` (detector) |
| a_lens (lens radius) | 1.5 mm | `aperture` |
| σ₀ (blur floor) | 0.12 mm | `blur_floor` |
| working distance | 5.0 mm | `working_distance` |
| α range | [1, 300] | fixed box in the model |
| focal lengths | 6–15 mm, 7 values cycled | lens records |
| array | 37 lenses, hex, pitch 3.6 mm | `hex_layout(rings=3, pitch=3.6)` |
