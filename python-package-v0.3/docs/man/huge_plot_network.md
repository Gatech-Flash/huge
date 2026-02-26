# `huge_plot_network`

## Usage

```python
huge_plot_network(
    fit,
    index=-1,
    ax=None,
    layout="spring",
    with_labels=False,
    node_size=120.0,
    node_color="#c44e52",
    edge_color="#4d4d4d",
    min_abs_weight=0.0,
) -> matplotlib.axes.Axes
```

## Description

Node-edge network visualization for one graph on the estimated path.

## Key arguments

- `layout`: `"spring"`, `"kamada_kawai"`, `"circular"`, `"spectral"`, `"shell"`
- `min_abs_weight`: drop weak edges with `abs(weight) < threshold`

## Dependencies

- `matplotlib`
- `networkx`
