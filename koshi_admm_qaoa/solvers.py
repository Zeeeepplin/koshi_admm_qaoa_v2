"""
solvers.py
==========
Solvers for the switch QUBO, plus a fixed-topology SOC reference search.

  solve_exact_qubo   - NumPyMinimumEigensolver (global QUBO minimum, small n)
  solve_qaoa         - QAOA on a noiseless (StatevectorSampler) or noisy (Aer)
                       backend; sweepable reps p
  solve_qrao         - Quantum Random Access Optimization (qubit-compressed)
  solve_sa_qubo      - classical simulated annealing on the QUBO
  brute_force_soc    - enumerate switch configs, score each by the corrected
                       fixed-topology SOC relaxation, and return the best radial
                       and connected configurations found

Every final returned solution is checked by power_model, including topology,
SOC diagnostics, and nonlinear AC recovery. The SOC relaxation is not treated
as a nonlinear-AC reference value.
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
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import RealAmplitudes

from network_data import Network
from qubo_builder import qubo_energy, z_to_dict
import power_model as pm


def set_algorithm_seed(seed: int) -> None:
    """Set Qiskit's algorithm-level RNG in addition to primitive seeds."""
    from qiskit_algorithms.utils import algorithm_globals

    algorithm_globals.random_seed = int(seed)


def variational_initial_point(seed: int, n_parameters: int) -> np.ndarray:
    """Predeclared seeded initial point in the ansatz's parameter order."""
    return np.random.default_rng(int(seed)).uniform(
        -np.pi, np.pi, size=int(n_parameters)
    )


# --------------------------- exact -----------------------------------------
def solve_exact_qubo(qp: QuadraticProgram):
    t = time.time()
    res = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    return {"method": "exact", "x": res.x.astype(int), "fval": float(res.fval),
            "time_s": time.time() - t, "n_qubits": qp.get_num_binary_vars(),
            "objective_symbol": "f_Q", "objective_sense": "minimize",
            "solution_scope": "one global QUBO minimizer returned",
            "returned_bits_order": "QuadraticProgram variable order"}


# --------------------------- QAOA ------------------------------------------
def _make_sampler(kind, shots, seed, noise_1q, noise_2q):
    # qiskit-algorithms 0.3 uses the Sampler V1 result contract
    # (``quasi_dists``). The Qiskit 1.2 StatevectorSampler/SamplerV2 contract
    # is intentionally not used here.
    if kind == "noiseless":
        return Sampler(
            options={"shots": int(shots), "seed": int(seed)}
        )
    from qiskit_aer.primitives import Sampler as AerSampler
    run_options = {"shots": int(shots), "seed": int(seed)}
    transpile_options = {"optimization_level": 0, "seed_transpiler": int(seed)}
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(noise_1q, 1),
                                   ["rz", "sx", "x", "h", "ry", "rx", "id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(noise_2q, 2),
                                   ["cx", "cz", "rzz", "ecr"])
    return AerSampler(
        backend_options={"noise_model": nm},
        transpile_options=transpile_options,
        run_options=run_options,
    )


def solve_qaoa(qp, reps=1, kind="noiseless", shots=2048, maxiter=80, seed=42,
               noise_1q=0.001, noise_2q=0.02, optimizer_tol=1.0e-4):
    set_algorithm_seed(seed)
    sampler = _make_sampler(kind, shots, seed, noise_1q, noise_2q)
    initial_point = variational_initial_point(seed, 2 * reps)
    trace = []

    def callback(eval_count, parameters, mean, metadata):
        if isinstance(metadata, dict):
            callback_metadata = {
                str(key): (
                    value
                    if isinstance(value, (str, int, float, bool, type(None)))
                    else str(value)
                )
                for key, value in metadata.items()
            }
        else:
            callback_metadata = {"value": str(metadata)}
        trace.append(
            {
                "evaluation": int(eval_count),
                "parameters": np.asarray(parameters, dtype=float).tolist(),
                "mean": float(np.real(mean)),
                "metadata": callback_metadata,
            }
        )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter, tol=optimizer_tol),
        reps=reps,
        initial_point=initial_point,
        callback=callback,
    )
    t = time.time()
    res = MinimumEigenOptimizer(qaoa).solve(qp)
    solver_result = res.min_eigen_solver_result
    optimal_point = getattr(solver_result, "optimal_point", None)
    return {"method": f"qaoa_p{reps}_{kind}", "x": res.x.astype(int),
            "fval": float(res.fval), "time_s": time.time() - t,
            "n_qubits": qp.get_num_binary_vars(), "reps": reps, "kind": kind,
            "objective_symbol": "f_Q", "objective_sense": "minimize",
            "hamiltonian_target": "minimum eigenvalue",
            "binary_to_pauli_mapping": "z_ell = (1 - Z_ell) / 2",
            "returned_bits_order": "QuadraticProgram variable order",
            "seed": int(seed), "shots": int(shots), "optimizer": "COBYLA",
            "optimizer_max_iterations": int(maxiter),
            "optimizer_tolerance": float(optimizer_tol),
            "initial_point": initial_point.tolist(),
            "optimal_point": (
                np.asarray(optimal_point, dtype=float).tolist()
                if optimal_point is not None else None
            ),
            "optimizer_trace": trace,
            "noise_1q": float(noise_1q) if kind != "noiseless" else 0.0,
            "noise_2q": float(noise_2q) if kind != "noiseless" else 0.0}


# --------------------------- QRAO -----------------------------------------
def solve_qrao(qp, max_vars_per_qubit=3, maxiter=60, shots=2048, seed=42,
               optimizer_tol=1.0e-4):
    from collections import defaultdict

    from qiskit_optimization.algorithms.qrao import (
        QuantumRandomAccessOptimizer, QuantumRandomAccessEncoding, MagicRounding)

    class MemoryBoundMagicRounding(MagicRounding):
        """MagicRounding with deterministic circuit batches.

        Qiskit Optimization 0.6.1 constructs every distinct magic-basis
        circuit before sampling. With 4096 shots and an uncompressed
        13-qubit encoding this can consume the workstation's full memory.
        Batching changes only primitive job boundaries; all requested bases
        and shot counts are retained.
        """

        def __init__(self, *args, batch_size=256, seed=None, **kwargs):
            self.batch_size = int(batch_size)
            self.batch_seed = int(seed) if seed is not None else 0
            super().__init__(*args, seed=seed, **kwargs)

        def _evaluate_magic_bases(
            self, circuit, bases, basis_shots, vars_per_qubit
        ):
            circuit_indices_by_shots = defaultdict(list)
            basis_counts = [None] * len(bases)
            if len(bases) != len(basis_shots):
                raise ValueError(
                    "number of magic bases and shot allocations differ"
                )
            for index, allocated_shots in enumerate(basis_shots):
                circuit_indices_by_shots[int(allocated_shots)].append(index)

            batch_index = 0
            for allocated_shots, indices in sorted(
                circuit_indices_by_shots.items(), reverse=True
            ):
                for start in range(0, len(indices), self.batch_size):
                    selected = indices[start : start + self.batch_size]
                    circuits = self._make_circuits(
                        circuit, bases[selected], vars_per_qubit
                    )
                    result = self.sampler.run(
                        circuits,
                        shots=allocated_shots,
                        seed=self.batch_seed + batch_index,
                    ).result()
                    counts_list = [
                        distribution.binary_probabilities()
                        for distribution in result.quasi_dists
                    ]
                    if len(counts_list) != len(selected):
                        raise RuntimeError(
                            "magic-rounding sampler returned the wrong "
                            "number of circuit results"
                        )
                    for index, counts in zip(selected, counts_list):
                        basis_counts[index] = counts
                    batch_index += 1
            if any(counts is None for counts in basis_counts):
                raise RuntimeError("magic-rounding circuit result is missing")
            return [
                {
                    key: value * int(basis_shots[index])
                    for key, value in counts.items()
                }
                for index, counts in enumerate(basis_counts)
            ]

    enc = QuantumRandomAccessEncoding(max_vars_per_qubit=max_vars_per_qubit)
    enc.encode(qp)
    set_algorithm_seed(seed)
    ansatz = RealAmplitudes(max(1, enc.num_qubits), reps=1)
    initial_point = variational_initial_point(seed, ansatz.num_parameters)
    trace = []

    def callback(eval_count, parameters, value, metadata):
        trace.append(
            {
                "evaluation": int(eval_count),
                "parameters": np.asarray(parameters, dtype=float).tolist(),
                "value": float(value),
                "metadata": str(metadata),
            }
        )

    vqe = VQE(
        Estimator(options={"shots": None}),
        ansatz,
        COBYLA(maxiter=maxiter, tol=optimizer_tol),
        initial_point=initial_point,
        callback=callback,
    )
    qrao = QuantumRandomAccessOptimizer(
        min_eigen_solver=vqe, max_vars_per_qubit=max_vars_per_qubit,
        rounding_scheme=MemoryBoundMagicRounding(
            sampler=Sampler(options={"shots": int(shots), "seed": int(seed)}),
            batch_size=256,
            seed=int(seed),
        ))
    t = time.time()
    res = qrao.solve(qp)
    rounding_samples = []
    for sample in getattr(res, "samples", []) or []:
        rounding_samples.append(
            {
                "bits_variable_order": np.asarray(sample.x, dtype=int).tolist(),
                "objective": float(sample.fval),
                "probability": float(sample.probability),
                "status": str(sample.status),
            }
        )
    q2vars = [[int(variable) for variable in group] for group in enc.q2vars]
    var2op = {}
    for variable, (qubit, operator) in enc.var2op.items():
        try:
            labels = operator.paulis.to_labels()
            pauli = labels[0] if len(labels) == 1 else labels
        except (AttributeError, TypeError):
            pauli = str(operator)
        var2op[str(int(variable))] = {"qubit": int(qubit), "pauli": pauli}
    return {"method": f"qrao_{max_vars_per_qubit}v", "x": res.x.astype(int),
            "fval": float(res.fval), "time_s": time.time() - t,
            "n_qubits": int(enc.num_qubits), "n_vars": qp.get_num_binary_vars(),
            "compression": qp.get_num_binary_vars() / max(1, enc.num_qubits),
            "objective_symbol": "f_Q", "objective_sense": "minimize",
            "relaxed_hamiltonian_target": "minimum eigenvalue",
            "returned_bits_order": "QuadraticProgram variable order",
            "encoding_q2vars": q2vars, "encoding_var2op": var2op,
            "encoding_offset": float(enc.offset),
            "seed": int(seed), "shots": int(shots), "optimizer": "COBYLA",
            "optimizer_max_iterations": int(maxiter),
            "optimizer_tolerance": float(optimizer_tol),
            "initial_point": initial_point.tolist(), "optimizer_trace": trace,
            "rounding": "MagicRounding",
            "rounding_implementation": (
                "Qiskit Optimization 0.6.1 MagicRounding with deterministic "
                "memory-bounded primitive batches"
            ),
            "rounding_sampler_batch_size": 256,
            "rounding_seed_policy": (
                "basis RNG uses trial seed; primitive batch j uses seed+j"
            ),
            "rounding_samples": rounding_samples}


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
            "time_s": time.time() - t, "n_qubits": n,
            "objective_symbol": "f_Q", "objective_sense": "minimize",
            "returned_bits_order": "meta variable order"}


# --------------------- fixed-topology SOC reference ------------------------
def brute_force_soc(net: Network, meta: dict, faulted=None, max_eval=20000):
    """Enumerate configurations and rank feasible ones by SOC-relaxation loss."""
    sw = meta["switch_branches"]; n = len(sw)
    best_rad = {"loss_mw": np.inf}; best_con = {"loss_mw": np.inf}
    n_eval = 0; t = time.time()
    for bits in itertools.product([0, 1], repeat=n):
        if n_eval >= max_eval:
            break
        z = np.array(bits); zd = z_to_dict(z, meta)
        if not pm.is_connected(net, zd, faulted):
            continue
        res = pm.solve_socp_fixed(net, zd, faulted)
        n_eval += 1
        if not res["soc_feasible"] or res["shed_mw"] > 1e-3:
            continue
        rec = {"z": z, "zd": zd, "loss_mw": res["loss_mw"],
               "qubo": qubo_energy(z, meta), "radial": pm.is_radial(net, zd, faulted)}
        if rec["loss_mw"] < best_con["loss_mw"]:
            best_con = rec
        if rec["radial"] and rec["loss_mw"] < best_rad["loss_mw"]:
            best_rad = rec
    return {"best_radial": best_rad, "best_connected": best_con,
            "n_eval": n_eval, "time_s": time.time() - t}


def brute_force_true(net: Network, meta: dict, faulted=None, max_eval=20000):
    """Backward-compatible alias; the result is a SOC reference, not AC truth."""
    return brute_force_soc(net, meta, faulted=faulted, max_eval=max_eval)


def score_config(net: Network, x: np.ndarray, meta: dict, faulted=None):
    """Run topology, corrected SOC, and nonlinear AC checks on a returned config."""
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
    gt = brute_force_soc(net, meta)
    print(f"exact QUBO fval={ex['fval']:.3f}  x={ex['x']}")
    bc = gt["best_connected"]; br = gt["best_radial"]
    print(f"SOC best CONNECTED loss={bc['loss_mw']:.3f} MW  radial={bc.get('radial')} "
          f"(evaluated {gt['n_eval']} connected configs in {gt['time_s']:.1f}s)")
    if np.isfinite(br["loss_mw"]):
        print(f"SOC best RADIAL loss={br['loss_mw']:.3f} MW")
    else:
        print("No fully-radial config exists for this switch subset (meshed operation).")
    sc = score_config(net, ex["x"], meta)
    print(f"exact-QUBO config -> SOC loss={sc['loss_mw']:.3f} MW shed={sc['shed_mw']:.2f} "
          f"radial={sc['radial']} connected={sc['connected']}")
