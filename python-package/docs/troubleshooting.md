# Troubleshooting

## First diagnostic command

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

If `runtime=True`, core deps are available.

## Full environment snapshot

```bash
pyhuge-doctor
```

## Symptom -> cause -> fix

## `PyHugeError: Native runtime for pyhuge is unavailable`

Cause: one or more runtime deps missing (`numpy`, `scipy`, `scikit-learn`).

Fix:

```bash
python -m pip install "pyhuge[runtime]"
```

## `scikit-learn is required for native mb/tiger` (or glasso)

Cause: `scikit-learn` is missing in current environment.

Fix:

```bash
python -m pip install scikit-learn
```

## Plotting import error (`matplotlib` / `networkx`)

Fix:

```bash
python -m pip install "pyhuge[viz]"
```

Headless environment:

```bash
export MPLBACKEND=Agg
```

## `error: externally-managed-environment` (PEP 668)

Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "pyhuge[runtime]"
```

## `index out of range` when plotting

Cause: `index` outside `fit.path` bounds.

Fix: use `index=-1` for the last graph or a valid index in `[0, len(fit.path)-1]`.

## `criterion='ebic' requires a glasso fit`

Cause: EBIC was requested for non-glasso estimator.

Fix: run `huge(..., method="glasso")` first, then `huge_select(..., criterion="ebic")`.
