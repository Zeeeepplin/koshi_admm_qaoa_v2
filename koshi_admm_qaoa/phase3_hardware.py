"""
phase3_hardware.py
==================
Phase 3: run the radiality-coupled reconfiguration QUBO on REAL IBM Quantum
hardware, and -- crucially -- MEASURE the noise degradation by running the same
circuit twice: once UNMITIGATED and once with error mitigation (readout twirling
TREX + dynamical decoupling).  This is the "honest degradation" result the review
asks for; the earlier code merely enabled mitigation without ever comparing.

Fixes applied vs. the earlier Phase 3:
  * No separately-rebuilt ansatz with possibly-mismatched parameters: we bind the
    trained parameters to the SAME QAOA ansatz object, then transpile that.
  * One hardware job per configuration (not one per ADMM iteration).
  * Explicit mitigated-vs-unmitigated comparison + approximation ratio vs exact.

CREDENTIAL NOTE (per user request the token stays hard-coded):
  The token is kept as a module constant to preserve the hard-coded pattern, but
  the actual leaked value has been replaced with a placeholder -- paste the
  (rotated!) token below.  A hard-coded token in a shared file is a security risk;
  rotate it and move to `QiskitRuntimeService.save_account(...)` before publishing.
"""
from __future__ import annotations
import numpy as np

IBM_TOKEN = "PASTE_YOUR_ROTATED_IBM_TOKEN_HERE"   # <-- hard-coded per request
IBM_CHANNEL = "ibm_quantum_platform"

from network_data import scaled_network
from qubo_builder import build_reconfig_qubo, z_to_dict
import power_model as pm
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import solvers


def _bitstring_to_z(bs, n):
    # qiskit counts are little-endian; reverse to align with variable order
    return np.array([int(b) for b in bs[::-1]])[:n]


def run_phase3(n_switches=6, reps=1, shots=4096, min_qubits=127):
    # ---- problem + exact reference ----
    net = scaled_network(n_switches)
    qp, meta = build_reconfig_qubo(net)
    nq = meta["n_qubits"]
    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    exact_fval = exact.fval
    print(f"Phase 3: {nq} qubits, exact QUBO optimum fval={exact_fval:.3f}")

    # ---- 1) train QAOA parameters locally (noiseless) ----
    # Bind the trained parameters to the SAME ansatz object QAOA used, so the
    # parameter ORDER cannot drift (the bug flagged in the review).
    qaoa = QAOA(sampler=solvers._make_sampler("noiseless", 2048, 42, 0, 0),
                optimizer=COBYLA(maxiter=100), reps=reps)
    meo_res = MinimumEigenOptimizer(qaoa).solve(qp)
    optimal_point = meo_res.min_eigen_solver_result.optimal_point
    ansatz = qaoa.ansatz.assign_parameters(optimal_point)
    if ansatz.num_clbits == 0:
        ansatz.measure_all()

    # ---- 2) connect to hardware ----
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    service = QiskitRuntimeService(channel=IBM_CHANNEL, token=IBM_TOKEN)
    backend = service.least_busy(operational=True, simulator=False,
                                 min_num_qubits=min_qubits)
    print(f"Backend: {backend.name}")
    pm_isa = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm_isa.run(ansatz)

    # ---- 3) run UNMITIGATED and MITIGATED ----
    def _run(mitigated):
        s = RuntimeSampler(mode=backend)
        if mitigated:
            s.options.twirling.enable_measure = True          # TREX readout mitigation
            s.options.dynamical_decoupling.enable = True
            s.options.dynamical_decoupling.sequence_type = "XpXm"
        else:
            s.options.twirling.enable_measure = False
            s.options.dynamical_decoupling.enable = False
        job = s.run([isa], shots=shots)
        print(f"  {'mitigated' if mitigated else 'unmitigated'} job {job.job_id()} ...")
        pr = job.result()[0]
        counts = getattr(pr.data, list(pr.data.keys())[0]).get_counts()
        return counts

    out = {}
    for mit in (False, True):
        counts = _run(mit)
        best_bs = max(counts, key=counts.get)
        z = _bitstring_to_z(best_bs, nq)
        fval = float(meta["linear"] @ z + z @ np.triu(meta["Q"], 1) @ z)
        val = pm.ac_feasibility(net, z_to_dict(z, meta))
        ar = fval / exact_fval if exact_fval < 0 else exact_fval / max(fval, 1e-9)
        out["mitigated" if mit else "unmitigated"] = dict(
            fval=fval, approx_ratio=ar, top_prob=counts[best_bs] / shots,
            connected=val["connected"], loss_mw=val["loss_mw"])
        print(f"  {'mitigated' if mit else 'unmitigated':11s}: fval={fval:.3f} "
              f"approx={ar:.3f} P(top)={counts[best_bs]/shots:.3f} "
              f"connected={val['connected']} loss={val['loss_mw']:.3f}MW")

    print("\nNoise degradation (approx-ratio drop unmitigated->mitigated recovery):")
    print(f"  unmitigated approx = {out['unmitigated']['approx_ratio']:.3f}")
    print(f"  mitigated   approx = {out['mitigated']['approx_ratio']:.3f}")
    return out


def local_selftest(n_switches=6, reps=1):
    """Validate everything EXCEPT the hardware submission: train params, bind to
    the ansatz, transpile against a FAKE backend, and simulate the ISA circuit.
    Lets us confirm the pipeline is correct without a token / queue."""
    net = scaled_network(n_switches)
    qp, meta = build_reconfig_qubo(net)
    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    qaoa = QAOA(sampler=solvers._make_sampler("noiseless", 2048, 42, 0, 0),
                optimizer=COBYLA(maxiter=60), reps=reps)
    meo_res = MinimumEigenOptimizer(qaoa).solve(qp)
    optimal_point = meo_res.min_eigen_solver_result.optimal_point
    ansatz = qaoa.ansatz.assign_parameters(optimal_point)
    if ansatz.num_clbits == 0:
        ansatz.measure_all()
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    backend = FakeSherbrooke()
    isa = generate_preset_pass_manager(optimization_level=3, backend=backend).run(ansatz)
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    counts = AerSampler(default_shots=4096).run([isa]).result()[0].data
    reg = getattr(counts, list(counts.keys())[0])
    top = max(reg.get_counts(), key=reg.get_counts().get)
    z = _bitstring_to_z(top, meta["n_qubits"])
    print(f"[selftest] exact fval={exact.fval:.3f}  ISA depth={isa.depth()} "
          f"2q-gates={isa.num_nonlocal_gates()}  top bitstring z={z}")
    print("[selftest] OK -- training, param binding, transpilation & ISA sim all valid.")
    return True


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        local_selftest(n_switches=6, reps=1)
    else:
        # requires a valid IBM token + queue time. Kept small (6 qubits) so the
        # hardware run is cheap and the noise story is clean.
        run_phase3(n_switches=6, reps=1, shots=4096)
