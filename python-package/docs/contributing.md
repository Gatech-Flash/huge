# Contributing

See `/Users/tourzhao/Desktop/huge-master/python-package/CONTRIBUTING.md` for
development environment setup and workflows.

Quick commands:

```bash
cd python-package
pip install -e ".[dev]"
pytest
mkdocs serve
```

Run optional e2e tests with local R runtime:

```bash
export R_LIBS_USER=/Users/tourzhao/Desktop/huge-master/.Rlib
R CMD INSTALL /Users/tourzhao/Desktop/huge-master
pytest tests/test_e2e_optional.py -rA
```

Release commands:

```bash
python scripts/bump_version.py 0.2.0
bash scripts/build_dist.sh
```

See `release.md` for full publishing flow.
