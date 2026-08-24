# Rekin interface (for agents)

Rekin is the constraint solver / global optimizer RIA delegates to
(`pip install rekin`). This file specifies its **interface only** — the model
file format, the CLI, the Python API, and what comes back. It says nothing
about how the solver works internally, and nothing here depends on that.

## 1. The `.rn` model file

Plain text. `#` starts a comment (whole-line or trailing). Blank line between
the variable block and the rest is conventional.

```
Variables:
  Real <name> : [<lo>, <hi>]      # continuous variable with box bounds
  Int  <name> : [<lo>, <hi>]      # integer variable

Define <name> := (<expression>)   # parse-time macro, see below

Constraints:
  <expr> <rel> <expr> ~ <tol>     # rel: =  !=  <  <=  >  >=
  <boolean combination>            # And / Or / Not / Implies over relations

Minimize:                          # or Maximize: — at most one objective
  <expression> ~ <tol>
```

- **Variables.** Every variable must be declared with finite box bounds. The
  solver searches the whole box; there is no initial guess anywhere in the
  format.
- **`Define name := (expr)`** is a parse-time macro: later occurrences of
  `name` are replaced textually by the parenthesised expression before
  parsing. Macros may reference earlier macros. No auxiliary variable is
  created — a chain of Defines produces one closed-form expression over the
  declared variables, while the file stays readable.
- **`~ tol`** (optional, per constraint and per objective) declares the
  tolerance within which that line counts as satisfied. Omitted → the
  problem default. Example: `x^2 + y^2 = 1 ~ 1e-6`.
- **Constraints section may be empty** (header alone is fine) — then the
  problem is pure box-constrained optimisation.
- **Expressions.** Operators `+ - * / ^` (power); functions
  `sin cos tan asin acos atan atan2 sinh cosh tanh atanh exp log sqrt abs`.
  Numbers in ordinary or scientific notation (`1e-9`).
- A file with no objective is a feasibility problem (find any point
  satisfying the constraints).

Minimal complete example:

```
Variables:
  Real x : [0, 5]

Constraints:
  x^2 = 2 ~ 1e-6
```

## 2. Command line

```bash
rekin model.rn --budget 30 --stall 5 --json
```

- `--budget S` — wall-clock solve budget, seconds.
- `--stall S`  — additionally stop after `S` seconds without improvement.
- `--json`     — machine-readable result on stdout (use this).

JSON fields an agent should read:

| key | meaning |
|---|---|
| `status_name` | `delta-global`, `delta-sat`, `local`, `best-effort`, `unsat`, `infeasible` |
| `obj` | objective value at the returned point (`null` for feasibility-only) |
| `sol` | `{variable_name: value}` — the returned point |
| `viol` | max constraint violation at the point |
| `time_s` | solve time |
| `time_to_best` | when the returned incumbent was first found |

## 3. Python API

```python
import rekin

# path or source text — same entry either way
r = rekin.solve("model.rn", budget=30, live=False)
r = rekin.solve(SOURCE_STRING, budget=30, live=False)

# or build programmatically
p = rekin.Problem(delta=1e-6)            # default tolerance
x = p.addVar("x", [0, 5])                # Real; addVar(..., type=int) for Int
y = p.addVar("y", [-5, 5])
p.addConstraint(x*x + y*y == 2, tol=1e-6)
p.addObjective(x + y)                    # or p.addObjective(expr, "max")
r = p.solve(budget=30, live=False)
```

`r` is a dict with the same keys as the CLI JSON (`r["status_name"]`,
`r["obj"]`, `r["sol"]["x"]`, `r["time_s"]`, ...). Expression builders for the
`Problem` API: `rekin.sin cos tan exp log sqrt tanh` plus Python arithmetic
on variables.

## 4. Statuses — how to read them

| status | claim |
|---|---|
| `delta-global` | global optimality proved to the declared tolerance |
| `delta-sat` | a point satisfying every constraint to tolerance (feasibility problems) |
| `local` | a certified local optimum |
| `best-effort` | best point found in the budget; no certificate |
| `unsat` / `infeasible` | no feasible point exists (proved) |

Statuses are never rounded up. When writing tests, assert the *class* of
status you expect (e.g. "delta-global or local"), not exact objective bytes —
`best-effort` incumbents can vary run to run within tolerance.
