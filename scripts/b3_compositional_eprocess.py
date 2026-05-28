"""
Compositional e-process empirical demo.

Build the (eps^*_t)_{t>=1} stream by ordering the
1,876 random-assignment ensemble bets by (seed, clique) and running
the bounded-difference e-process at three lambdas, plotting
E_t over time. Also report which lambdas cross 1/alpha = 10^4 and
when.

Theoretical envelope. Under H_0 (all components coherent), we use
m^* / (4 * min_a K_{a,t}) for the conditional mean and
m^* / (2 * min_a K_{a,t}) for the variance proxy.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402

OUT_JSON = REPO_ROOT / "results" / "b3_compositional_eprocess.json"
OUT_FIG = REPO_ROOT / "figures" / "b3_eprocess_compositional.pdf"
SEED = 0
SEEDS = 4
N_MODELS = 4
KEEP = ("neg", "and", "or", "partition")
K_PER_SAMPLE = 8


def make_clique(p: np.ndarray, relation: str) -> Clique:
    m = p.size
    if relation == "partition":
        rels = [Relation(type="partition", indices=tuple(range(m)))]
    elif relation == "neg":
        rels = [Relation(type="neg", indices=(0, 1))]
    elif relation == "and":
        rels = [Relation(type="and", indices=(0, 1, 2))]
    elif relation == "or":
        rels = [Relation(type="or", indices=(0, 1, 2))]
    else:
        raise ValueError(relation)
    return Clique(m=m, relations=rels, p_hat=p)


def build_eps_star_stream() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (eps^*_t, m^*_t, K_t) stream from the ensemble."""
    npz = np.load(REPO_ROOT / "figures" / "master_combined.npz", allow_pickle=True)
    with open(REPO_ROOT / "figures" / "master_combined.json") as f:
        meta_json = json.load(f)
    relations_arr = np.array(meta_json["relations"])
    n_total = len(relations_arr)
    n_cliques = n_total // N_MODELS
    forecast_jcd = npz["forecast__JCD"].reshape(N_MODELS, n_cliques, -1)
    sizes_full = npz["clique_sizes"].reshape(N_MODELS, n_cliques)
    rels_full = relations_arr.reshape(N_MODELS, n_cliques)
    rels = rels_full[0]
    sizes = sizes_full[0]
    keep_idx = np.where(np.isin(rels, KEEP))[0]

    rng = np.random.default_rng(SEED)
    eps_t, m_t, K_t = [], [], []
    for seed in range(SEEDS):
        for ci in keep_idx:
            relation = str(rels[ci])
            m = int(sizes[ci])
            assignment = rng.integers(0, N_MODELS, size=m)
            if relation == "partition":
                if any(int(sizes_full[a, ci]) < m for a in assignment):
                    continue
            composed = np.array(
                [forecast_jcd[assignment[j], ci, j] for j in range(m)]
            )
            clique = make_clique(composed, relation)
            proj = jcd_project(clique)
            eps_t.append(float(np.linalg.norm(composed - proj)))
            m_t.append(m)
            K_t.append(K_PER_SAMPLE)
    return np.array(eps_t), np.array(m_t), np.array(K_t)


def e_process(eps_t: np.ndarray, m_t: np.ndarray, K_t: np.ndarray,
              lam: float) -> np.ndarray:
    """E_t(lambda) = prod_{s<=t} exp(lam*(eps_s^2 - m_s/(4 K_s))
                                    - lam^2 m_s / (2 K_s))."""
    eps2 = eps_t ** 2
    centered = eps2 - m_t / (4.0 * K_t)
    variance_term = (lam ** 2) * m_t / (2.0 * K_t)
    log_factor = lam * centered - variance_term
    return np.exp(np.cumsum(log_factor))


def main() -> None:
    eps_t, m_t, K_t = build_eps_star_stream()
    print(f"Compositional residual stream: T = {len(eps_t)} bets")
    print(f"  mean eps^*: {eps_t.mean():.4f}, max: {eps_t.max():.4f}")
    print(f"  H_0 envelope mean[eps^*^2]: m^*/(4K), avg = {(m_t / (4 * K_t)).mean():.4f}")

    # Run e-process at multiple lambdas; mix-of-lambdas approach (Ramdas 2023).
    lambdas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    E_curves = {lam: e_process(eps_t, m_t, K_t, lam) for lam in lambdas}
    # Mixture e-process (uniform over the grid): retains anytime-validity.
    E_mix = np.mean(np.stack([E_curves[lam] for lam in lambdas]), axis=0)

    # When does each lambda cross 1/alpha = 10^4?
    threshold = 1e4
    crossings = {}
    for lam in lambdas:
        idx = np.where(E_curves[lam] >= threshold)[0]
        crossings[lam] = int(idx[0]) + 1 if idx.size > 0 else None
    mix_idx = np.where(E_mix >= threshold)[0]
    crossings["mixture"] = int(mix_idx[0]) + 1 if mix_idx.size > 0 else None

    print("\nLambda thresholds:")
    for lam in lambdas:
        c = crossings[lam]
        print(f"  lam={lam}: crosses 10^4 at t={c}" if c else f"  lam={lam}: no crossing in T={len(eps_t)}")
    print(f"  mixture: crosses 10^4 at t={crossings['mixture']}")

    # Save numeric output.
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(
            T=len(eps_t),
            mean_eps_star=float(eps_t.mean()),
            crossings=crossings,
            lambdas=lambdas,
            mixture_E_final=float(E_mix[-1]),
        ), f, indent=2)

    # Plot.
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    t = np.arange(1, len(eps_t) + 1)
    for lam in lambdas:
        ax.plot(t, E_curves[lam], label=f"$\\lambda={lam}$", alpha=0.6, lw=1.0)
    ax.plot(t, E_mix, "k-", label="mixture", lw=2.0)
    ax.axhline(threshold, color="r", ls="--", label="$1/\\alpha=10^4$")
    ax.set_yscale("log")
    ax.set_xlabel("ensemble bet $t$ (random-assignment order)")
    ax.set_ylabel("$E_t$")
    ax.set_title("Compositional e-process on $(\\epsilon^\\star_t)$, $T=%d$ bets" % len(eps_t))
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f"\nWritten {OUT_JSON}")
    print(f"Written {OUT_FIG}")


if __name__ == "__main__":
    main()
