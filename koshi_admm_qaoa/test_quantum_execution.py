"""Small execution-contract checks for every quantum solver path.

These tests deliberately use four binary variables, two optimizer evaluations,
and 64 shots.  They are not performance experiments; they guard against API
drift that would leave an advertised solver path non-executable.
"""

import numpy as np

from network_data import scaled_network
from qubo_builder import build_reconfig_qubo, qubo_energy
from solvers import solve_exact_qubo, solve_qaoa, solve_qrao


def test_exact_qaoa_and_qrao_paths_execute_and_report_consistent_objectives():
    qp, meta = build_reconfig_qubo(scaled_network(4, seed=0))

    exact = solve_exact_qubo(qp)
    qaoa = solve_qaoa(qp, shots=64, maxiter=2, seed=7)
    qrao = solve_qrao(qp, shots=64, maxiter=2, seed=7)
    qrao_repeat = solve_qrao(qp, shots=64, maxiter=2, seed=7)

    for result in (exact, qaoa, qrao):
        bits = np.asarray(result["x"], dtype=int)
        assert bits.shape == (4,)
        assert np.isclose(result["fval"], qubo_energy(bits, meta))
        assert result["objective_sense"] == "minimize"

    assert len(qaoa["optimizer_trace"]) == 2
    assert len(qrao["optimizer_trace"]) == 2
    assert qrao["rounding_samples"]
    assert qrao["rounding_samples"] == qrao_repeat["rounding_samples"]
    assert np.array_equal(qrao["x"], qrao_repeat["x"])
    assert qrao["rounding_sampler_batch_size"] == 256
