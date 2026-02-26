"""Minimal pyhuge example: MB path estimation + RIC selection."""

import numpy as np

from pyhuge import huge, huge_npn, huge_select


def main() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(150, 40))

    x_t = huge_npn(x, npn_func="shrinkage", verbose=False)
    fit = huge(x_t, method="mb", nlambda=8, lambda_min_ratio=0.1, verbose=False)
    sel = huge_select(fit, criterion="ric", verbose=False)

    print("method:", fit.method)
    print("num lambda:", len(fit.lambda_path))
    print("selected lambda:", sel.opt_lambda)
    print("selected sparsity:", sel.opt_sparsity)


if __name__ == "__main__":
    main()
