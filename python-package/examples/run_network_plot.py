"""Network visualization example (node/edge plot)."""

import matplotlib.pyplot as plt
import numpy as np

from pyhuge import huge_mb, huge_plot_network


def main() -> None:
    rng = np.random.default_rng(29)
    x = rng.normal(size=(120, 24))
    fit = huge_mb(x, nlambda=6, verbose=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    huge_plot_network(fit, index=-1, ax=ax, layout="spring", with_labels=False)
    fig.tight_layout()
    fig.savefig("pyhuge_network_plot.png", dpi=150)
    print("saved: pyhuge_network_plot.png")


if __name__ == "__main__":
    main()
