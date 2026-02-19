"""Minimal pyhuge example: graphical lasso + EBIC selection."""

import numpy as np

from pyhuge import huge, huge_select


def main() -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(120, 30))

    fit = huge(
        x,
        method="glasso",
        nlambda=10,
        lambda_min_ratio=0.1,
        cov_output=False,
        verbose=False,
    )
    sel = huge_select(fit, criterion="ebic", ebic_gamma=0.5, verbose=False)

    print("method:", fit.method)
    print("loglik shape:", None if fit.loglik is None else fit.loglik.shape)
    print("selected lambda:", sel.opt_lambda)
    print("refit shape:", sel.refit.shape)


if __name__ == "__main__":
    main()
