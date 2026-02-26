# Installation

## Recommended

```bash
pip install "pyhuge[runtime]"
```

Source install:

```bash
git clone https://github.com/Gatech-Flash/huge.git
cd huge/python-package-v0.3
pip install -e ".[runtime]"
```

## Optional extras

```bash
pip install -e ".[viz]"   # matplotlib + networkx
pip install -e ".[test]"  # pytest
pip install -e ".[docs]"  # mkdocs
pip install -e ".[dev]"   # common contributor setup
```

## Verify install

```bash
python -c "import pyhuge; print(pyhuge.test())"
pyhuge-doctor
```

`runtime=True` means core dependencies are available.

## Apple Silicon / architecture notes

`pyhuge` 0.3 does not depend on R architecture, but Python/native wheels must
match your interpreter architecture. Check:

```bash
python -c 'import platform,sys; print(platform.machine()); print(sys.executable)'
```

## PEP 668 environments (externally-managed)

If `pip install` shows `externally-managed-environment`, use a virtual env:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "pyhuge[runtime]"
```
