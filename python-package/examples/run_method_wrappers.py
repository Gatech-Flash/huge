"""Method-specific wrapper example."""

import numpy as np

from pyhuge import huge_ct, huge_glasso, huge_mb, huge_tiger


def main() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 20))

    mb = huge_mb(x, nlambda=5, verbose=False)
    ct = huge_ct(x, nlambda=5, verbose=False)
    glasso = huge_glasso(x, nlambda=5, verbose=False)
    tiger = huge_tiger(x, nlambda=5, verbose=False)

    print("mb:", len(mb.path), mb.method)
    print("ct:", len(ct.path), ct.method)
    print("glasso:", len(glasso.path), glasso.method)
    print("tiger:", len(tiger.path), tiger.method)


if __name__ == "__main__":
    main()
