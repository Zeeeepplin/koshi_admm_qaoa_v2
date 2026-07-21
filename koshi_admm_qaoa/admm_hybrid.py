"""
admm_hybrid.py
==============
Corrected hybrid quantum-classical ADMM for post-contingency transmission
switching on the Eastern-Nepal sub-system.

Decomposition (consensus ADMM):
  x-update (CLASSICAL, convex AC):  alpha = argmin  losses + shed
                                    + (rho/2)||alpha - z + u||^2
                                    s.t. SOC branch-flow (power_model.solve_socp_admm_x)
  z-update (QUANTUM/combinatorial):  z = argmin  radiality-coupled QUBO
                                     (qubo_builder.build_reconfig_qubo with the
                                      consensus target alpha, dual u, weight rho)
                                     solved by QAOA (or exact eigensolver).
  y-update:  u <- u + (alpha - z)

vs. the original code this fixes the two things that made the quantum step
pointless: the z-update QUBO is now COUPLED (radiality), and the loop is driven
by a real AC SOCP x-update on a real meshed network.  Residual history is logged
and the final binary topology is AC-validated.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np

from network_data import Network, build_full_network
from qubo_builder import build_reconfig_qubo, z_to_dict
import power_model as pm
import solvers


def run_admm(net: Network, rho: float = 3.0, max_iter: int = 30,
             eps_primal: float = 1e-2, eps_dual: float = 1e-2,
             z_solver: str = "qaoa", qaoa_reps: int = 1, qaoa_kind: str = "noiseless",
             faulted=None, verbose: bool = True):
    sw = net.switch_indices(); n = len(sw)
    z = np.ones(n); z_prev = z.copy(); u = np.zeros(n)
    hist = {"primal": [], "dual": [], "loss": [], "shed": [], "alpha": [], "z": []}
    t0 = time.time()
    if verbose:
        print(f"ADMM on {net.name}: {n} switches, rho={rho}, z-solver={z_solver}"
              f"{'/p'+str(qaoa_reps)+'/'+qaoa_kind if z_solver=='qaoa' else ''}")
    final = None
    for it in range(max_iter):
        # ---- x-update: convex AC SOCP with consensus penalty ----
        alpha, loss, shed = pm.solve_socp_admm_x(net, z, u, rho, faulted)

        # ---- z-update: radiality-coupled QUBO via QAOA / exact ----
        # Meshed transmission operation: consensus-led (no hard spanning-tree
        # cardinality); loop + anti-islanding terms supply the coupling.
        qp, meta = build_reconfig_qubo(net, alpha=alpha, u=u, rho=rho,
                                       lambda_card=0.0,
                                       lambda_cycle=0.4 * rho,
                                       lambda_iso=0.8 * rho,
                                       loss_bias=0.3 * rho)
        if z_solver == "qaoa":
            r = solvers.solve_qaoa(qp, reps=qaoa_reps, kind=qaoa_kind, maxiter=60)
        else:
            r = solvers.solve_exact_qubo(qp)
        z_prev = z.copy(); z = r["x"].astype(float)

        # ---- y-update ----
        u = u + (alpha - z)
        primal = float(np.linalg.norm(alpha - z))
        dual = float(np.linalg.norm(-rho * (z - z_prev)))
        hist["primal"].append(primal); hist["dual"].append(dual)
        hist["loss"].append(loss); hist["shed"].append(shed)
        hist["alpha"].append(np.round(alpha, 3).tolist()); hist["z"].append(z.astype(int).tolist())
        if verbose:
            print(f"  it{it+1:02d}  primal={primal:.3e} dual={dual:.3e} "
                  f"loss={loss:6.3f}MW shed={shed:5.2f}MW  z={z.astype(int)}")
        if primal < eps_primal and dual < eps_dual:
            if verbose:
                print(f"  converged at iteration {it+1}")
            break

    # ---- connectivity repair + validate final topology on the TRUE AC model ----
    zd_raw = z_to_dict(z.astype(int), meta)
    zd = pm.connectivity_repair(net, zd_raw)
    repaired = zd != zd_raw
    val = pm.ac_feasibility(net, zd, faulted)
    z_final = np.array([zd[k] for k in sw])
    final = {"z": z_final, "z_dict": zd, "z_raw": z.astype(int), "repaired": repaired,
             "iters": len(hist["primal"]), "time_s": time.time() - t0,
             "validation": val, "history": hist, "switch_branches": sw}
    if verbose:
        tag = " (after connectivity repair)" if repaired else ""
        print(f"  FINAL z={z_final}{tag}  AC: loss={val['loss_mw']:.3f}MW "
              f"shed={val['shed_mw']:.2f}MW connected={val['connected']} "
              f"radial={val['radial']} vmin={val['vmin_pu']:.3f}pu feasible={val['feasible']}")
    return final


def plot_convergence(result, path="figures/admm_convergence.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = result["history"]; it = range(1, len(h["primal"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(it, h["primal"], "o-", label="primal residual ||α−z||")
    ax[0].semilogy(it, h["dual"], "s-", label="dual residual ||ρ(z−z⁻)||")
    ax[0].set_xlabel("ADMM iteration"); ax[0].set_ylabel("residual (log)")
    ax[0].set_title("ADMM convergence"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(it, h["loss"], "o-", color="tab:green", label="AC losses (MW)")
    ax[1].plot(it, h["shed"], "s-", color="tab:red", label="load shed (MW)")
    ax[1].set_xlabel("ADMM iteration"); ax[1].set_ylabel("MW")
    ax[1].set_title("Objective components"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == "__main__":
    net = build_full_network()
    # exact z-solver first (fast, verifies the loop math), then QAOA
    print("=== ADMM with EXACT z-update ===")
    res_exact = run_admm(net, rho=3.0, z_solver="exact", max_iter=30)
    print("\n=== ADMM with QAOA (p=1, noiseless) z-update ===")
    res_qaoa = run_admm(net, rho=3.0, z_solver="qaoa", qaoa_reps=1, max_iter=15)
    p = plot_convergence(res_qaoa)
    print("saved", p)
