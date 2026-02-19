"""Summary and plotting helper example."""

import matplotlib.pyplot as plt
import numpy as np

from pyhuge import (
    huge_glasso,
    huge_plot_graph_matrix,
    huge_plot_sparsity,
    huge_select,
    huge_select_summary,
    huge_summary,
)


def main() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(100, 25))

    fit = huge_glasso(x, nlambda=8, verbose=False)
    sel = huge_select(fit, criterion="ebic", verbose=False)

    print(huge_summary(fit))
    print(huge_select_summary(sel))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    huge_plot_sparsity(fit, ax=axes[0])
    huge_plot_graph_matrix(fit, index=-1, ax=axes[1])
    fig.tight_layout()
    fig.savefig("pyhuge_summary_plot.png", dpi=150)
    print("saved: pyhuge_summary_plot.png")


if __name__ == "__main__":
    main()
