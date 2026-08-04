"""Regression tests for the frozen experimental design."""
from __future__ import annotations

from study_protocol import build_protocol, validate_protocol


def test_retained_artifact_is_explicitly_no_contingency():
    protocol = build_protocol()
    validate_protocol(protocol)
    legacy = protocol["retained_legacy_protocol"]
    assert legacy["contingency"] is None
    assert legacy["faulted_branches"] == []
    assert legacy["interpretation"] == "base case; not post-contingency"


def test_prospective_contingency_order_and_statistics_are_predeclared():
    primary = build_protocol()["prospective_primary_protocol"]
    assert primary["status"].startswith("predeclared")
    assert primary["contingency"]["forced_open_branch_index"] == 3
    assert primary["decision_set"]["n_binary_variables"] == 13
    order = primary["decision_set"]["variable_order"]
    assert [item["position"] for item in order] == list(range(13))
    assert 3 not in [item["branch_index"] for item in order]
    assert primary["topology_precheck"]["spanning_tree_feasible"]
    assert primary["statistical_analysis"][
        "independent_runs_per_stochastic_method"
    ] == 30
    assert primary["statistical_analysis"]["bootstrap_resamples"] == 10000