# Tutorials

Runnable examples are in `python-package-v0.3/examples/`.

## Installation check

```bash
cd python-package-v0.3
python -c "import pyhuge; print(pyhuge.test())"
```

## Example scripts

- `examples/run_huge_mb.py`
  - MB path estimation + RIC selection
- `examples/run_huge_glasso.py`
  - glasso path estimation + EBIC selection
- `examples/run_method_wrappers.py`
  - wrapper shortcuts (`huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`)
- `examples/run_sim_inference_roc.py`
  - simulation, ROC, and inference workflow
- `examples/run_summary_and_plot.py`
  - summary helpers and matrix/sparsity plots
- `examples/run_network_plot.py`
  - node-edge graph visualization (`huge_plot_network`)
- `examples/run_native_demo.py`
  - compact end-to-end native demo

## Typical local run

```bash
cd python-package-v0.3
python examples/run_huge_mb.py
```

## Optional full e2e checks

```bash
cd python-package-v0.3
pytest tests/test_e2e_optional.py tests/test_rd_examples_parity.py -rA
```
