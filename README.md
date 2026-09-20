# pycg-ml

ML-aware static call graphs for Python.

pycg-ml is a fork of [PyCG](https://github.com/vitsalis/PyCG) — Practical Python Call
Graphs (Salis et al., ICSE 2021). Upstream was archived on 26 November 2023 and no
longer runs on current Python: since 3.12 `importlib.invalidate_caches()` imports part
of the standard library lazily, which collides with the global import hook PyCG installs
while analysing a package, and the run aborts.

This fork exists to keep that analysis usable and to teach it the patterns that tabular
ML pipelines are built from — feature-engineering steps, estimators and their
composition. It is the call-graph backend for research on MPFI, a static metric of
structural fragility in ML pipelines.

## Status

- Runs on Python 3.12 and 3.13.
- Reproduces upstream's results exactly: 106 of 119 micro-benchmark snippets pass,
  the same 13 that upstream fails on Python 3.10 (starred assignment, `map`/`types`
  builtins, decorators, dicts, `eval`, list slices, MRO/`super`).
- Recovers the call edges tabular ML code declares as data rather than writing as
  calls: a transform handed to `DataFrame.pipe`, the steps of an sklearn `Pipeline`,
  `ColumnTransformer` or `FeatureUnion`, and evaluation functions and callbacks passed
  to lightgbm or xgboost. `--no-ml-patterns` returns the plain upstream graph, which
  keeps the two readings comparable for ablation.

## Install

```sh
uv sync
```

## Use

```sh
uv run pycg-ml --package path/to/package path/to/package/entry.py -o callgraph.json
```

## Develop

```sh
make test              # unit tests, including the Python 3.12+ import regressions
make functional-test   # the 119-snippet gold standard
make lint              # ruff + mypy
```

## Licence and attribution

Apache License 2.0, inherited from upstream. PyCG is Copyright (c) 2020 Vitalis Salis;
changes made in this fork are listed in [NOTICE](NOTICE) and are not endorsed by the
original authors.

If you use this work academically, cite the original paper:

> Vitalis Salis, Thodoris Sotiropoulos, Panos Louridas, Diomidis Spinellis, Dimitris
> Mitropoulos. PyCG: Practical Call Graph Generation in Python. ICSE 2021.
