"""Simulation example: generator + huge + roc + inference."""

from pyhuge import huge, huge_generator, huge_inference, huge_roc


def main() -> None:
    sim = huge_generator(n=120, d=20, graph="hub", g=4, verbose=False)
    fit = huge(sim.data, method="glasso", nlambda=6, verbose=False)

    roc = huge_roc(fit.path, sim.theta, verbose=False, plot=False)
    inf = huge_inference(
        data=sim.data,
        t=fit.icov[-1],
        adj=sim.theta,
        alpha=0.05,
        type_="Gaussian",
    )

    print("AUC:", roc.auc)
    print("Inference type-I error:", inf.error)


if __name__ == "__main__":
    main()
