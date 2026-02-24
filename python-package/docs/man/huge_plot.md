# `huge_plot`

## Usage

```python
huge_plot(
    g,
    epsflag=False,
    graph_name="default",
    cur_num=1,
    location=None,
) -> Optional[str]
```

## Description

Python wrapper for R `huge.plot()` style graph visualization.

## Key arguments

- `g`: square adjacency matrix
- `epsflag`: if `True`, save EPS file
- `graph_name`: EPS filename prefix
- `cur_num`: number suffix in output name
- `location`: output directory when `epsflag=True`

## Returns

- `None` when `epsflag=False`
- EPS file path when `epsflag=True`
