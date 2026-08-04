"""Run both ADMM variants, archive raw histories, and generate the figure."""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from network_data import build_full_network
from admm_hybrid import run_admm


ROOT = Path(__file__).resolve().parent


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main():
    net = build_full_network()
    exact = run_admm(net, rho=4.0, z_solver="exact", max_iter=25, verbose=False)
    qaoa = run_admm(
        net, rho=4.0, z_solver="qaoa", qaoa_reps=1, max_iter=12, verbose=False
    )

    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "rho": 4.0,
            "exact_max_iterations": 25,
            "qaoa_max_iterations": 12,
            "qaoa_depth": 1,
        },
        "exact": _jsonable(exact),
        "qaoa": _jsonable(qaoa),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "admm_runs.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    he, hq = exact["history"], qaoa["history"]
    axes[0].semilogy(
        range(1, len(he["primal"]) + 1), he["primal"], "o-", label="exact z-update"
    )
    axes[0].semilogy(
        range(1, len(hq["primal"]) + 1), hq["primal"], "s--", label="QAOA z-update"
    )
    axes[0].set_xlabel("ADMM iteration")
    axes[0].set_ylabel("primal residual ||alpha-z|| (log)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, len(he["loss"]) + 1), he["loss"], "o-", label="exact")
    axes[1].plot(range(1, len(hq["loss"]) + 1), hq["loss"], "s--", label="QAOA")
    axes[1].set_xlabel("ADMM iteration")
    axes[1].set_ylabel("fixed-topology SOC loss (MW)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "admm_convergence.png", dpi=160)
    plt.close(fig)
    print("wrote results/admm_runs.json and figures/admm_convergence.png")


if __name__ == "__main__":
    main()
