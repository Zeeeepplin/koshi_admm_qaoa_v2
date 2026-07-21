"""
solvers.py
==========
Solvers for the radiality-coupled switch QUBO, plus a TRUE-objective ground truth.

  solve_exact_qubo   - NumPyMinimumEigensolver (exact QUBO optimum, small n)
  solve_qaoa         - QAOA on a noiseless (StatevectorSampler) or noisy (Aer)
                       backend; sweepable reps p
  solve_qrao         - Quantum Random Access Optimization (qubit-compressed)
  solve_sa_qubo      - classical simulated annealing on the QUBO
  brute_force_true   - enumerate switch configs, score each by the TRUE AC SOCP
                       loss (power_model), returns best radial & best connected

Every returned solution is re-scored with power_model so the benchmark compares
methods on the REAL objective (AC losses + feasibility), not just the surrogate.
"""
from __future__ import annotations
import time, itertools, math, random, warnings
from typing import Dict, Optional, Tuple
import numpy as np
warnings.filterwarnings("ignore")

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.circuit.library import RealAmplitudes

from network_data import Network
from qubo_builder import qubo_energy, z_to_dict
import power_model as pm


# --------------------------- exact -----------------------------------------
def solve_exact_qubo(qp: QuadraticProgram):
    t = time.time()
    res = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    return {"method": "exact", "x": res.x.astype(int), "fval": float(res.fval),
            "time_s": time.time() - t, "n_qubits": qp.get_num_binary_vars()}


# --------------------------- QAOA ------------------------------------------
class _TranspilingSampler:
    """Wrap an Aer SamplerV2 so incoming QAOA circuits (which contain a
    high-level PauliEvolution 'QAOA' gate) are decomposed into basis gates Aer
    understands.  Uses Aer for BOTH noiseless and noisy runs, which scales to
    ~20 qubits -- StatevectorSampler densifies the evolution operator (O(4^n))
    and blows up beyond ~11 qubits."""
    _BASIS = ["rz", "sx", "x", "cx", "id", "rzz", "h", "ry", "rx", "cz"]

    def __init__(self, aer):
        self._aer = aer
        self._cache = {}

    def run(self, pubs, *, shots=None):
        from qiskit import transpile
        new = []
        for pub in pubs:
            circ = pub[0]
            key = id(circ)
            if key not in self._cache:
                self._cache[key] = transpile(circ, basis_gates=self._BASIS,
                                             optimization_level=0)
            new.append((self._cache[key],) + tuple(pub[1:]))
        return self._aer.run(new, shots=shots) if shots is not None else self._aer.run(new)


def _make_sampler(kind, shots, seed, noise_1q, noise_2q):
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    if kind == "noiseless":
        aer = AerSampler(default_shots=shots, seed=seed,
                         options={"backend_options": {"method": "statevector"}})
        return _TranspilingSampler(aer)
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(noise_1q, 1),
                                   ["rz", "sx", "x", "h", "ry", "rx", "id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(noise_2q, 2),
                                   ["cx", "cz", "rzz", "ecr"])
    aer = AerSampler(default_shots=shots, seed=seed,
                     options={"backend_options": {"noise_model": nm}})
    return _TranspilingSampler(aer)


def solve_qaoa(qp, reps=1, kind="noiseless", shots=2048, maxiter=80, seed=42,
               noise_1q=0.001, noise_2q=0.02):
    sampler = _make_sampler(kind, shots, seed, noise_1q, noise_2q)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=reps)
    t = time.time()
    res = MinimumEigenOptimizer(qaoa).solve(qp)
    return {"method": f"qaoa_p{reps}_{kind}", "x": res.x.astype(int),
            "fval": float(res.fval), "time_s": time.time() - t,
            "n_qubits": qp.get_num_binary_vars(), "reps": reps, "kind": kind}


# --------------------------- QRAO -----------------------------------------
def solve_qrao(qp, max_vars_per_qubit=3, maxiter=60, shots=2048, seed=42):
    from qiskit_optimization.algorithms.qrao import (
        QuantumRandomAccessOptimizer, QuantumRandomAccessEncoding, MagicRounding)
    enc = QuantumRandomAccessEncoding(max_vars_per_qubit=max_vars_per_qubit)
    enc.encode(qp)
    vqe = VQE(StatevectorEstimator(),
              RealAmplitudes(max(1, enc.num_qubits), reps=1), COBYLA(maxiter=maxiter))
    qrao = QuantumRandomAccessOptimizer(
        min_eigen_solver=vqe, max_vars_per_qubit=max_vars_per_qubit,
        rounding_scheme=MagicRounding(sampler=StatevectorSampler(default_shots=shots, seed=seed)))
    t = time.time()
    res = qrao.solve(qp)
    return {"method": f"qrao_{max_vars_per_qubit}v", "x": res.x.astype(int),
            "fval": float(res.fval), "time_s": time.time() - t,
            "n_qubits": int(enc.num_qubits), "n_vars": qp.get_num_binary_vars(),
            "compression": qp.get_num_binary_vars() / max(1, enc.num_qubits)}


# --------------------------- Simulated annealing ---------------------------
def solve_sa_qubo(meta, iters=400, T0=10.0, cooling=0.98, seed=42):
    rng = random.Random(seed)
    n = meta["n_qubits"]
    cur = np.array([rng.randint(0, 1) for _ in range(n)])
    cur_e = qubo_energy(cur, meta); best, best_e = cur.copy(), cur_e
    T = T0; t = time.time()
    for _ in range(iters):
        i = rng.randrange(n); nb = cur.copy(); nb[i] ^= 1
        e = qubo_energy(nb, meta); d = e - cur_e
        if d < 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
            cur, cur_e = nb, e
            if e < best_e:
                best, best_e = nb.copy(), e
        T *= cooling
    return {"method": "sa", "x": best.astype(int), "fval": float(best_e),
            "time_s": time.time() - t, "n_qubits": n}


# --------------------------- TRUE ground truth -----------------------------
def brute_force_true(net: Network, meta: dict, faulted=None, max_eval=20000):
    """Score every switch config by TRUE AC SOCP loss. Returns best radial/connected."""
    sw = meta["switch_branches"]; n = len(sw)
    best_rad = {"loss_mw": np.inf}; best_con = {"loss_mw": np.inf}
    n_eval = 0; t = time.time()
    for bits in itertools.product([0, 1], repeat=n):
        if n_eval >= max_eval:
            break
        z = np.array(bits); zd = z_to_dict(z, meta)
        if not pm.is_connected(net, zd):
            continue
        res = pm.solve_socp_fixed(net, zd, faulted)
        n_eval += 1
        if not res["feasible"] or res["shed_mw"] > 1e-3:
            continue
        rec = {"z": z, "zd": zd, "loss_mw": res["loss_mw"],
               "qubo": qubo_energy(z, meta), "radial": pm.is_radial(net, zd)}
        if rec["loss_mw"] < best_con["loss_mw"]:
            best_con = rec
        if rec["radial"] and rec["loss_mw"] < best_rad["loss_mw"]:
            best_rad = rec
    return {"best_radial": best_rad, "best_connected": best_con,
            "n_eval": n_eval, "time_s": time.time() - t}


def score_config(net: Network, x: np.ndarray, meta: dict, faulted=None):
    """Evaluate a method's returned config on the TRUE objective."""
    zd = z_to_dict(np.asarray(x).astype(int), meta)
    res = pm.ac_feasibility(net, zd, faulted)
    res["qubo"] = qubo_energy(np.asarray(x).astype(int), meta)
    return res


if __name__ == "__main__":
    from network_data import scaled_network
    from qubo_builder import build_reconfig_qubo
    net = scaled_network(8)
    qp, meta = build_reconfig_qubo(net)
    ex = solve_exact_qubo(qp)
    gt = brute_force_true(net, meta)
    print(f"exact QUBO fval={ex['fval']:.3f}  x={ex['x']}")
    bc = gt["best_connected"]; br = gt["best_radial"]
    print(f"TRUE best CONNECTED loss={bc['loss_mw']:.3f} MW  radial={bc.get('radial')} "
          f"(evaluated {gt['n_eval']} connected configs in {gt['time_s']:.1f}s)")
    if np.isfinite(br["loss_mw"]):
        print(f"TRUE best RADIAL loss={br['loss_mw']:.3f} MW")
    else:
        print("No fully-radial config exists for this switch subset (meshed operation).")
    sc = score_config(net, ex["x"], meta)
    print(f"exact-QUBO config -> TRUE loss={sc['loss_mw']:.3f} MW shed={sc['shed_mw']:.2f} "
          f"radial={sc['radial']} connected={sc['connected']}")
