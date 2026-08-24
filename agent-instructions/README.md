# agent-instructions — machine-oriented reference

These files are written for LLM agents (not humans) that want to build on RIA:
generate new benchmark instances, write new `.rn` models in the same style, or
drive the solver directly. They specify **interfaces and file formats only**.

| file | contents |
|---|---|
| [`rekin-interface.md`](rekin-interface.md) | the solver's exact input format (`.rn`), CLI, Python API, result fields and statuses |
| [`ria-encoding.md`](ria-encoding.md) | how RIA encodes the imaging likelihood as an `.rn` model, macro by macro |
| [`making-benchmarks.md`](making-benchmarks.md) | recipes and rules for generating new, correct benchmark instances |

Ground truth for everything here: the seven examples in `../examples/` and the
emitted models in `../examples/models/`. When these documents and a real
emitted model disagree, the emitted model wins — regenerate it by running the
example (`keep_model=` writes the exact file that was solved).
