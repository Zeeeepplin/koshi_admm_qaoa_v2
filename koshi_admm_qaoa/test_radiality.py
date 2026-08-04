"""Regression tests for the exact radiality correction."""
from __future__ import annotations

import numpy as np
import pytest

from network_data import build_full_network, scaled_network
from qubo_builder import build_reconfig_qubo
from radiality import (
    fixed_forest_status,
    project_to_spanning_tree,
    solve_exact_enumeration,
    solve_spanning_tree_milp,
    solve_uniform_cardinality_qubo,
    surrogate_radiality_audit,
)


def test_full_network_fixed_subgraph_is_a_forest_and_milp_finds_tree():
    network = build_full_network()
    fixed = fixed_forest_status(network)
    assert fixed["fixed_is_forest"]
    assert fixed["target_closed_switches"] == 8

    result = solve_spanning_tree_milp(network)
    assert result["success"]
    assert result["connected"]
    assert result["radial"]
    assert result["closed_branch_count"] == network.n_bus - 1


def test_projection_opens_cycles_and_returns_a_tree():
    network = build_full_network()
    all_closed = {index: 1 for index in network.switch_indices()}
    result = project_to_spanning_tree(network, all_closed)
    assert result["success"]
    assert result["radial"]
    assert result["opened_switches"]
    assert not result["closed_switches"]
    assert result["n_switch_changes"] == len(result["opened_switches"])


def test_exact_enumeration_archives_complete_minimizer_set():
    network = scaled_network(4)
    _, meta = build_reconfig_qubo(network, build_program=False)
    result = solve_exact_enumeration(meta)
    assert result["success"]
    assert result["minimizer_count"] == len(
        result["all_minimizers_variable_order"]
    )
    assert result["x"].tolist() in result["all_minimizers_variable_order"]


@pytest.mark.parametrize("n", [4, 6, 8])
def test_legacy_small_scaling_instances_cannot_be_radial(n):
    network = scaled_network(n)
    assert not fixed_forest_status(network)["fixed_is_forest"]
    result = solve_spanning_tree_milp(network)
    assert not result["success"]
    assert "mandatory fixed branches contain a cycle" in result["status"]


@pytest.mark.parametrize("n", [10, 12])
def test_larger_scaling_instances_have_exact_radial_solutions(n):
    result = solve_spanning_tree_milp(scaled_network(n))
    assert result["success"]
    assert result["radial"]


@pytest.mark.parametrize("n", [4, 6, 8, 10, 12])
def test_sorting_prefix_solver_matches_enumeration_on_easy_variant(n):
    network = scaled_network(n)
    _, easy = build_reconfig_qubo(
        network,
        lambda_cycle=0.0,
        lambda_iso=0.0,
        build_program=False,
    )
    prefix = solve_uniform_cardinality_qubo(easy)
    exact = solve_exact_enumeration(easy)
    assert np.isclose(prefix["objective"], exact["objective"])


def test_prefix_solver_rejects_actual_nonuniform_qubo():
    _, actual = build_reconfig_qubo(scaled_network(12), build_program=False)
    with pytest.raises(ValueError, match="nonuniform"):
        solve_uniform_cardinality_qubo(actual)


def test_pairwise_cycle_term_has_false_positives_and_false_negatives():
    network = scaled_network(12)
    _, meta = build_reconfig_qubo(network, build_program=False)
    audit = surrogate_radiality_audit(network, meta)
    assert audit["cycle_surrogate_false_positive_radial_states"] > 0
    assert (
        audit["cycle_surrogate_false_negative_nonradial_target_cardinality_states"]
        > 0
    )
    assert audit["global_qubo_minimizers_that_are_radial"] == 0


def test_predeclared_contingency_is_excluded_from_qubo_decision_vector():
    network = build_full_network()
    _, meta = build_reconfig_qubo(network, faulted=[3], build_program=False)
    assert meta["n_qubits"] == 13
    assert 3 not in meta["switch_branches"]
    assert meta["faulted_branches"] == [3]
    assert meta["topology_encoding"][
        "faulted_branches_excluded_from_decision_vector"
    ] == [3]
    result = solve_spanning_tree_milp(network, faulted=[3])
    assert result["success"]
