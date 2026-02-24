# Tutorials

`pyhuge` ships runnable example scripts under `python-package/examples/`.

## Installation check

```bash
cd python-package
python -c "import pyhuge; print(pyhuge.test())"
```

If runtime dependencies are available (`rpy2`, R, and R package `huge`), `runtime`
will be `True`.

## Example scripts

- `examples/run_huge_mb.py`
  - neighborhood selection (`mb`) path estimation
- `examples/run_huge_glasso.py`
  - graphical lasso path estimation and model selection
- `examples/run_method_wrappers.py`
  - method-specific wrappers (`huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`)
- `examples/run_sim_inference_roc.py`
  - simulation, ROC, and inference workflow
- `examples/run_summary_and_plot.py`
  - summaries, sparsity curve, and matrix visualization
- `examples/run_network_plot.py`
  - node-edge network visualization (`huge_plot_network`)

## Typical local runtime command

```bash
cd python-package
R_LIBS_USER=/path/to/Rlib \
python examples/run_huge_mb.py
```

## Optional e2e test run

```bash
cd python-package
export R_LIBS_USER=/path/to/Rlib
export PYHUGE_REQUIRE_RUNTIME=1
pytest tests/test_e2e_optional.py -rA
```
