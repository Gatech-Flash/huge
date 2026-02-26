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

Native huge-style graph visualization helper.

## Key arguments

- `g`: square adjacency matrix
- `epsflag`: if `True`, save EPS file
- `graph_name`: filename prefix
- `cur_num`: positive integer suffix
- `location`: output directory when `epsflag=True`

## Returns

- `None` when `epsflag=False`
- EPS file path when `epsflag=True`
