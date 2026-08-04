"""
admm_hybrid.py
==============
Hybrid quantum-classical ADMM-inspired consensus heuristic for
post-contingency transmission switching on the Eastern-Nepal sub-system.

Decomposition (consensus ADMM):
  x-update (CLASSICAL, convex AC):  alpha = argmin  losses + shed
                                    + (rho/2)||alpha - z + u||^2
                                    s.t. SOC branch-flow (power_model.solve_socp_admm_x)
  z-update (QUANTUM/combinatorial):  z = argmin  heuristic topology QUBO
                                     (qubo_builder.build_reconfig_qubo with the
                                      consensus target alpha, dual u, weight rho)
                                     solved by QAOA (or exact eigensolver).
  y-update:  u <- u + (alpha - z)

The finite topology penalties do not make this loop an exact ADMM
decomposition of the hard connected-radial problem. Residual history is logged,
and the raw terminal iterate is kept separate from a post-hoc spanning-tree
projection and its physical validation.
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
             qaoa_seed: int = 42, qaoa_maxiter: int = 60,
             qaoa_optimizer_tol: float = 1.0e-4,
             faulted=None, verbose: bool = True):
    faulted = sorted({int(index) for index in (faulted or [])})
    sw = [index for index in net.switch_indices() if index not in set(faulted)]
    n = len(sw)
    z = np.ones(n); z_prev = z.copy(); u = np.zeros(n)
    hist = {
        "primal": [], "dual": [], "loss": [], "shed": [], "alpha": [], "z": [],
        "z_qubo_objective": [], "z_solver_time_s": [],
    }
    t0 = time.time()
    if verbose:
        print(f"ADMM on {net.name}: {n} switches, rho={rho}, z-solver={z_solver}"
              f"{'/p'+str(qaoa_reps)+'/'+qaoa_kind if z_solver=='qaoa' else ''}")
    final = None
    termination_reason = "maximum_iterations"
    for it in range(max_iter):
        # ---- x-update: convex AC SOCP with consensus penalty ----
        alpha, loss, shed = pm.solve_socp_admm_x(net, z, u, rho, faulted)

        # ---- z-update: heuristic topology-coupled QUBO via QAOA / exact ----
        # Meshed transmission operation: consensus-led (no hard spanning-tree
        # cardinality); loop + anti-islanding terms supply the coupling.
        qp, meta = build_reconfig_qubo(net, alpha=alpha, u=u, rho=rho,
                                       lambda_card=0.0,
                                       lambda_cycle=0.4 * rho,
                                       lambda_iso=0.8 * rho,
                                       loss_bias=0.3 * rho,
                                       faulted=faulted)
        if z_solver == "qaoa":
            r = solvers.solve_qaoa(
                qp,
                reps=qaoa_reps,
                kind=qaoa_kind,
                maxiter=qaoa_maxiter,
                seed=qaoa_seed,
                optimizer_tol=qaoa_optimizer_tol,
            )
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
        hist["z_qubo_objective"].append(float(r["fval"]))
        hist["z_solver_time_s"].append(float(r["time_s"]))
        if verbose:
            print(f"  it{it+1:02d}  primal={primal:.3e} dual={dual:.3e} "
                  f"loss={loss:6.3f}MW shed={shed:5.2f}MW  z={z.astype(int)}")
        if primal < eps_primal and dual < eps_dual:
            termination_reason = "primal_and_dual_tolerances"
            if verbose:
                print(f"  converged at iteration {it+1}")
            break

    # ---- exact connected-radial projection + SOC/nonlinear validation ----
    zd_raw = z_to_dict(z.astype(int), meta)
    from qubo_builder import qubo_energy
    from radiality import topology_status

    raw_topology = topology_status(net, zd_raw, faulted)
    raw_qubo_objective = qubo_energy(z.astype(int), meta)
    repair = pm.radiality_repair(net, zd_raw, faulted)
    zd = repair["z"]
    repaired = bool(repair.get("n_switch_changes", 0))
    val = pm.ac_feasibility(net, zd, faulted)
    z_final = np.array([zd[k] for k in sw])
    projected_qubo_objective = qubo_energy(z_final.astype(int), meta)
    final = {"z": z_final, "z_dict": zd, "z_raw": z.astype(int), "repaired": repaired,
             "raw_topology": raw_topology,
             "raw_qubo_objective": float(raw_qubo_objective),
             "projected_qubo_objective": float(projected_qubo_objective),
             "projected_objective_change": float(
                 projected_qubo_objective - raw_qubo_objective
             ),
             "repair": repair,
             "iters": len(hist["primal"]), "time_s": time.time() - t0,
             "validation": val, "history": hist, "switch_branches": sw,
             "termination_reason": termination_reason,
             "configuration": {
                 "rho": rho, "max_iterations": max_iter,
                 "primal_tolerance": eps_primal, "dual_tolerance": eps_dual,
                 "z_solver": z_solver, "qaoa_reps": qaoa_reps,
                 "qaoa_kind": qaoa_kind, "qaoa_seed": qaoa_seed,
                 "qaoa_max_iterations": qaoa_maxiter,
                 "qaoa_optimizer_tolerance": qaoa_optimizer_tol,
                 "faulted_branches": faulted,
                 "algorithm_classification": (
                     "ADMM-inspired consensus heuristic with finite topology "
                     "penalties and post-hoc spanning-tree projection"
                 ),
             }}
    if verbose:
        tag = " (after spanning-tree projection)" if repaired else ""
        print(f"  FINAL z={z_final}{tag}  SOC: loss={val['loss_mw']:.3f}MW "
              f"shed={val['shed_mw']:.2f}MW connected={val['connected']} "
              f"radial={val['radial']} vmin={val.get('vmin_pu', float('nan')):.3f}pu "
              f"soc_feasible={val['soc_feasible']} "
              f"nonlinear_ac_validated={val['nonlinear_ac_validated']}")
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
