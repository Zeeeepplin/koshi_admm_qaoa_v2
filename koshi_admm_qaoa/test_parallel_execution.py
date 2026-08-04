"""Smoke-test the process-isolated prospective workers with tiny budgets."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from benchmark import _prospective_quantum_trial
from make_post_contingency_admm import (
    _active_switch_branches,
    _qaoa_admm_seed_run,
)
from network_data import build_full_network
from run_post_contingency_sensitivity import _sensitivity_trial
from study_protocol import build_protocol


def _tiny_primary():
    primary = deepcopy(build_protocol()["prospective_primary_protocol"])
    primary["qaoa"]["shots"] = 32
    primary["qaoa"]["optimizer_max_iterations"] = 1
    primary["qrao"]["shots"] = 32
    primary["qrao"]["optimizer_max_iterations"] = 1
    primary["admm"]["maximum_iterations"] = 1
    return primary


def test_process_isolated_workers_are_picklable_and_execute():
    primary = _tiny_primary()
    fopt = -999.0  # only the gap contract is exercised in this smoke test
    variant = {
        "name": "smoke",
        "qubo_parameters": {
            key: primary["qubo_parameters"][key]
            for key in ("lambda_card", "lambda_cycle", "lambda_iso", "loss_bias")
        },
        "qaoa_max_iterations": 1,
    }
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as pool:
        qaoa_future = pool.submit(
            _prospective_quantum_trial,
            ("qaoa", 1, 7, primary, fopt),
        )
        sensitivity_future = pool.submit(
            _sensitivity_trial,
            (variant, primary, 8, fopt),
        )
        qaoa_trial, qaoa_qubits = qaoa_future.result()
        sensitivity_trial = sensitivity_future.result()

    assert qaoa_qubits == primary["decision_set"]["n_binary_variables"]
    assert qaoa_trial["seed"] == 7
    assert sensitivity_trial["seed"] == 8
    assert qaoa_trial["raw_candidate"]["bits_variable_order"]
    assert sensitivity_trial["projection"]["projected_bits_variable_order"]

    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        admm_result = pool.submit(_qaoa_admm_seed_run, (primary, 9)).result()
    assert admm_result["configuration"]["qaoa_seed"] == 9
    assert len(admm_result["history"]["primal"]) == 1


def test_admm_artifact_uses_current_network_switch_api():
    network = build_full_network()
    faulted = [3]
    active = _active_switch_branches(network, faulted)

    assert active == [
        index for index in network.switch_indices() if index not in faulted
    ]
    assert 3 not in active
