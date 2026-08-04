"""Regression tests for raw-versus-projected prospective trial evidence."""
from __future__ import annotations

import numpy as np

import benchmark
from network_data import build_full_network
from qubo_builder import build_reconfig_qubo, qubo_energy


def test_trial_keeps_raw_and_projected_candidates_separate(monkeypatch):
    network = build_full_network()
    _, meta = build_reconfig_qubo(
        network, faulted=[3], build_program=False
    )
    bits = np.ones(meta["n_binary_variables"], dtype=int)
    objective = qubo_energy(bits, meta)
    raw_score = {
        "model_revision": "test",
        "soc_feasible": True,
        "connected": True,
        "radial": False,
        "nonlinear_ac_validated": False,
    }
    monkeypatch.setattr(
        benchmark.solvers,
        "score_config",
        lambda *args, **kwargs: {
            "model_revision": "test",
            "soc_feasible": True,
            "connected": True,
            "radial": True,
            "fixed_topology_soc_validated": True,
            "nonlinear_ac_validated": True,
        },
    )
    trial = benchmark._trial(
        1001,
        {
            "method": "test",
            "x": bits,
            "fval": objective,
            "time_s": 0.1,
            "n_qubits": len(bits),
        },
        raw_score,
        objective,
        network,
        meta,
        faulted=[3],
    )
    assert trial["raw_candidate"]["bits_variable_order"] == bits.tolist()
    assert trial["raw_candidate"]["topology"]["radial"] is False
    assert trial["projection"]["required"] is True
    assert trial["projection"]["projected_topology"]["radial"] is True
    assert (
        trial["projection"]["projected_bits_variable_order"]
        != trial["raw_candidate"]["bits_variable_order"]
    )
