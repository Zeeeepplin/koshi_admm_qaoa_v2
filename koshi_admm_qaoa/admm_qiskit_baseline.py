"""
admm_qiskit_baseline.py
=======================
Library baseline requested in the review: run Qiskit-Optimization's built-in
`ADMMOptimizer` (the validated 3-ADMM-H implementation) on a mixed binary/
continuous reconfiguration model, instead of only a hand-rolled loop.

HONEST SCOPE: `ADMMOptimizer`'s continuous block must be a convex QP/LP solved by
its `continuous_optimizer`; it CANNOT handle the second-order-cone AC branch-flow
constraints.  So this baseline uses a LINEARISED reconfiguration surrogate
(linear loss proxy + linear radiality/anti-islanding constraints + a continuous
coupling slack).  The full AC-SOC coupling remains in admm_hybrid.py.  Reporting
both — a library baseline on the linear surrogate and our SOC-accurate hand-rolled
loop — is exactly the comparison the review recommends.
"""
from __future__ import annotations
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (ADMMOptimizer, MinimumEigenOptimizer,
                                            CobylaOptimizer)
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA

from network_data import build_full_network, scaled_network
from qubo_builder import fundamental_cycles, _fixed_rank, _bus_incident_switch
import solvers


def build_admm_qp(net):
    """Mixed binary/continuous QP in ADMMOptimizer's sweet spot: a QUADRATIC
    (radiality-coupled) objective on the binary switches + one continuous
    coupling variable tied by a single linear EQUALITY.  The radiality/anti-
    islanding structure lives in the objective (soft, coupled -> QAOA-relevant),
    not in many hard inequalities that the library struggles to reconcile."""
    from qubo_builder import build_reconfig_qubo, z_to_dict  # reuse the coupled QUBO
    _, meta = build_reconfig_qubo(net, lambda_card=1.0, lambda_cycle=1.0,
                                  lambda_iso=3.0, loss_bias=3.0)
    sw = meta["switch_branches"]; n = len(sw)
    lin, Q = meta["linear"], meta["Q"]

    qp = QuadraticProgram("koshi_admm_coupled")
    for k in sw:
        qp.binary_var(name=f"z{k}")
    qp.continuous_var(lowerbound=0, upperbound=n, name="y")  # continuous block

    linear = {f"z{sw[i]}": float(lin[i]) for i in range(n)}
    linear["y"] = 0.10                                   # mild cost on the coupler
    quad = {(f"z{sw[i]}", f"z{sw[j]}"): float(Q[i, j])
            for i in range(n) for j in range(i + 1, n) if abs(Q[i, j]) > 1e-12}
    qp.minimize(linear=linear, quadratic=quad)
    # single equality couples the two blocks:  y = sum(z)
    qp.linear_constraint({**{f"z{k}": 1 for k in sw}, "y": -1}, "==", 0, name="couple")
    return qp, sw, meta


def run_qiskit_admm(net=None, use_qaoa=False, maxiter=6):
    net = net or scaled_network(8)
    qp, sw, meta = build_admm_qp(net)
    if use_qaoa:
        qubo_solver = MinimumEigenOptimizer(
            QAOA(sampler=solvers._make_sampler("noiseless", 1024, 42, 0, 0),
                 optimizer=COBYLA(maxiter=40), reps=1))
    else:
        qubo_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    # ADMMOptimizer needs a strong penalty to enforce the linear (in)equality
    # constraints; it is noticeably more sensitive than the hand-rolled SOC loop.
    params = ADMMParameters(rho_initial=10.0, beta=1.0, factor_c=10.0,
                            maxiter=maxiter, tol=1e-3, three_block=True)
    admm = ADMMOptimizer(qubo_optimizer=qubo_solver,
                         continuous_optimizer=CobylaOptimizer(),
                         params=params)
    res = admm.solve(qp)
    x = np.array(res.x[:len(sw)]).astype(int)
    # same connectivity-repair used for the hand-rolled loop -> fair comparison
    import power_model as pm
    from qubo_builder import z_to_dict
    zd = pm.connectivity_repair(net, z_to_dict(x, meta))
    val = pm.ac_feasibility(net, zd)
    xr = np.array([zd[k] for k in sw])
    print(f"Qiskit ADMMOptimizer on {net.name}: status={res.status.name} fval={res.fval:.3f}")
    print(f"  raw z={x}  -> repaired z={xr}")
    print(f"  AC (repaired): connected={val['connected']} loss={val['loss_mw']:.3f}MW "
          f"shed={val['shed_mw']:.2f}MW")
    return res, xr


if __name__ == "__main__":
    print("=== Qiskit ADMMOptimizer (exact QUBO block) ===")
    run_qiskit_admm(scaled_network(8), use_qaoa=False)
    print("\n=== Qiskit ADMMOptimizer (QAOA QUBO block) ===")
    run_qiskit_admm(scaled_network(8), use_qaoa=True)
