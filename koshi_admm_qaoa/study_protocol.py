"""Machine-readable experiment specification for the revised study.

The retained benchmark is a legacy *base-case* scaling artifact: no faulted
branch was passed to either the QUBO or continuous model.  It must not be cited
as post-contingency evidence.

The prospective primary study is predeclared here but has not been run.  It uses
a hypothetical N-1 outage of branch 3 (Basantapur--Inaruwa circuit A), excludes
that branch from the binary decision vector, and studies the full remaining
13-switch instance.  The outage is a reproducible scenario choice, not a claim
that a historical outage occurred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from network_data import build_full_network
from radiality import solve_spanning_tree_milp


PROTOCOL_SCHEMA_VERSION = 2
PROTOCOL_REVISION = "post-contingency-protocol-v2"
PRIMARY_CONTINGENCY_BRANCH = 3
PRIMARY_SEEDS = list(range(1001, 1031))
BOOTSTRAP_SEED = 20260724
ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "results" / "study_protocol.json"
PROTOCOL_TABLE_PATH = ROOT / "generated" / "study_protocol_table.tex"
SWITCH_ORDER_TABLE_PATH = ROOT / "generated" / "switch_order_table.tex"


def _sha256_record(record) -> str:
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _branch_record(branch, position=None, decision=True) -> dict:
    return {
        "position": position,
        "branch_index": int(branch.idx),
        "name": branch.name,
        "from_bus": int(branch.frm),
        "to_bus": int(branch.to),
        "kind": branch.kind,
        "decision_variable": bool(decision),
    }


def build_protocol() -> dict:
    network = build_full_network()
    switch_indices = network.switch_indices()
    historical_order = [
        _branch_record(network.branches[index], position=position)
        for position, index in enumerate(switch_indices)
    ]
    contingency = network.branches[PRIMARY_CONTINGENCY_BRANCH]
    primary_indices = [
        index for index in switch_indices if index != PRIMARY_CONTINGENCY_BRANCH
    ]
    primary_order = [
        _branch_record(network.branches[index], position=position)
        for position, index in enumerate(primary_indices)
    ]
    topology_check = solve_spanning_tree_milp(
        network, faulted=[PRIMARY_CONTINGENCY_BRANCH]
    )
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_revision": PROTOCOL_REVISION,
        "frozen_on": str(date(2026, 7, 25)),
        "scope": {
            "retained_artifact": (
                "legacy no-contingency surrogate-QUBO scaling audit"
            ),
            "prospective_primary_study": (
                "hypothetical N-1 post-contingency full-decision-set experiment"
            ),
            "important_rule": (
                "prospective results must be stored separately and may not replace "
                "the retained legacy artifact until the complete run is validated"
            ),
        },
        "notation_contract": {
            "physical_bus_indices": "i,j",
            "physical_branch_index": "k",
            "binary_vector_positions": "ell,m",
            "variable_order_map": "pi(ell) = physical branch index",
            "continuous_admm_copy": "alpha",
            "binary_admm_qubo_copy": "z",
            "returned_or_measured_bitstring": "b",
            "qubo_objective": "f_Q",
            "qubo_global_minimum": "f_Q_star",
            "qubo_minimizer_set": "Z_Q_star",
            "optimizer_run_hit_fraction": "h_run_star",
            "shot_level_exact_set_probability": "p_shot_star",
            "binary_to_pauli_mapping": "z_ell = (1 - Z_ell) / 2",
            "qaoa_objective_sense": "minimize Hamiltonian expectation",
        },
        "retained_legacy_protocol": {
            "contingency": None,
            "faulted_branches": [],
            "interpretation": "base case; not post-contingency",
            "sizes": [4, 6, 8, 10, 12],
            "variable_order_full": historical_order,
            "prefix_rule": (
                "the first n entries are binary decisions; all later switchable "
                "branches are forced closed"
            ),
            "known_design_failure": (
                "mandatory fixed subgraphs are cyclic at n=4,6,8"
            ),
            "qubo_parameters": {
                "lambda_card": 3.0,
                "lambda_cycle": 1.5,
                "lambda_iso": 6.0,
                "loss_bias": 5.0,
                "rho": 0.0,
            },
            "qaoa": {
                "depths": [1, 2],
                "shots": 1024,
                "optimizer": "COBYLA",
                "optimizer_max_iterations": 50,
                "optimizer_tolerance": "not explicitly set in legacy run",
                "seeds": [42, 7, 123],
                "initial_point": "not archived",
            },
            "qrao": {
                "retained_runs_per_size": 1,
                "seed": "not recoverable from aggregate artifact",
            },
            "simulated_annealing": {
                "iterations": 300,
                "initial_temperature": 10.0,
                "cooling_factor": 0.98,
                "seeds": [42, 7, 123],
            },
            "statistical_limit": (
                "three QAOA/SA seeds and one QRAO run; descriptive only"
            ),
        },
        "prospective_primary_protocol": {
            "status": "predeclared; no numerical results generated",
            "scenario_name": "hypothetical_n_minus_1_basantapur_inaruwa_ckt_a",
            "scenario_type": "hypothetical deterministic N-1 branch outage",
            "historical_event_claimed": False,
            "contingency": {
                "forced_open_branch_index": int(contingency.idx),
                "forced_open_branch_name": contingency.name,
                "from_bus": int(contingency.frm),
                "to_bus": int(contingency.to),
                "rationale": (
                    "outage of one circuit in a switchable parallel 220-kV pair; "
                    "the planned companion circuit is modeled as available in "
                    "this prospective test-system scenario"
                ),
            },
            "network": {
                "name": network.name,
                "n_bus": int(network.n_bus),
                "n_branch": int(network.n_branch),
                "continuous_model_revision": "continuous-model-v2",
                "topology_revision": "exact-radiality-v1",
                "network_status": (
                    "source-informed test system combining reported in-service "
                    "assets, planned circuits, and hypothetical interfaces; "
                    "not an as-operated NEA network model"
                ),
                "radiality_policy": (
                    "experimental emergency-restoration policy surrogate; "
                    "not a generic requirement of transmission OTS"
                ),
                "nonlinear_validation_scope": (
                    "series-only fixed-injection AC recovery; line charging, "
                    "shunts, magnetizing branches, and phase shifts are omitted"
                ),
            },
            "execution": {
                "parallel_workers": 4,
                "parallel_unit": "independent optimizer run",
                "ordering": "results stored in the predeclared seed order",
                "rng_isolation": (
                    "process-based workers isolate Qiskit's process-global "
                    "algorithm seed"
                ),
            },
            "decision_set": {
                "scaling_sweep": False,
                "n_binary_variables": len(primary_order),
                "variable_order": primary_order,
                "excluded_faulted_branch": int(contingency.idx),
                "reason_no_scaling_sweep": (
                    "avoid the invalid forced-closed prefix construction and make "
                    "all solver/physics comparisons use one decision space"
                ),
            },
            "topology_precheck": {
                "spanning_tree_feasible": bool(topology_check["success"]),
                "target_closed_switches": topology_check.get(
                    "target_closed_switches"
                ),
                "precheck_scope": "graph topology only; not SOC or nonlinear AC",
            },
            "qubo_parameters": {
                "lambda_card": 3.0,
                "lambda_cycle": 1.5,
                "lambda_iso": 6.0,
                "loss_bias": 5.0,
                "rho": 0.0,
                "interpretation": (
                    "historical heuristic objective retained for solver comparison; "
                    "hard radiality is assessed separately"
                ),
            },
            "qaoa": {
                "depths": [1, 2],
                "shots": 4096,
                "optimizer": "COBYLA",
                "optimizer_max_iterations": 100,
                "optimizer_tolerance": 1.0e-4,
                "initial_point_distribution": (
                    "independent Uniform[-pi,pi) values in Qiskit ansatz parameter order"
                ),
                "seed_controls": [
                    "initial-point generator",
                    "qiskit algorithm_globals",
                    "sampler/transpiler randomness where supported",
                ],
                "seeds": PRIMARY_SEEDS,
            },
            "qrao": {
                "max_variables_per_qubit": 3,
                "ansatz": "RealAmplitudes(reps=1)",
                "rounding": "MagicRounding",
                "shots": 4096,
                "optimizer": "COBYLA",
                "optimizer_max_iterations": 100,
                "optimizer_tolerance": 1.0e-4,
                "initial_point_distribution": (
                    "independent Uniform[-pi,pi) values in ansatz parameter order"
                ),
                "required_encoding_metadata": [
                    "q2vars qubit-to-variable groups",
                    "var2op variable-to-qubit-and-Pauli map",
                    "relaxed-Hamiltonian offset",
                    "returned bits in QuadraticProgram variable order",
                ],
                "seeds": PRIMARY_SEEDS,
            },
            "simulated_annealing": {
                "iterations": 1000,
                "initial_temperature": 10.0,
                "cooling_factor": 0.995,
                "seeds": PRIMARY_SEEDS,
            },
            "exact_qubo": {
                "method": "NumPyMinimumEigensolver via MinimumEigenOptimizer",
                "runs": 1,
                "reported_quantities": [
                    "global QUBO minimum f_Q_star",
                    "one returned minimizer in variable order",
                    "complete minimizer set when explicitly enumerated",
                ],
            },
            "admm": {
                "rho": 3.0,
                "rho_schedule": "constant",
                "maximum_iterations": 30,
                "primal_tolerance": 1.0e-2,
                "dual_tolerance": 1.0e-2,
                "initial_z": "all available switches closed",
                "initial_scaled_dual": "zero vector",
                "z_update_parameters": {
                    "lambda_card": 0.0,
                    "lambda_cycle": 1.2,
                    "lambda_iso": 2.4,
                    "loss_bias": 0.9,
                },
                "required_history": [
                    "primal residual at every iteration",
                    "dual residual at every iteration",
                    "continuous copy alpha and binary copy z at every iteration",
                    "loss and shedding at every iteration",
                    "termination reason and final repair actions",
                    "raw terminal topology and QUBO objective",
                    "projected topology, QUBO objective, and objective change",
                ],
            },
            "sensitivity_analysis": {
                "design": "one factor at a time around the primary setting",
                "multipliers": [0.5, 1.0, 2.0],
                "parameters": [
                    "lambda_card",
                    "lambda_cycle",
                    "lambda_iso",
                    "loss_bias",
                    "ADMM rho",
                ],
                "seeds": PRIMARY_SEEDS,
                "interpretation": "descriptive robustness; no causal attribution",
            },
            "statistical_analysis": {
                "independent_runs_per_stochastic_method": len(PRIMARY_SEEDS),
                "primary_estimands": [
                    "raw QUBO gap Delta_Q = f_Q(b) - f_Q_star",
                    "optimizer-run exact-QUBO hit indicator b in Z_Q_star",
                    "wall time",
                    "connected/radial indicators",
                    "current-model SOC and nonlinear-AC validation indicators",
                ],
                "continuous_summary": (
                    "median, interquartile range, and within-method percentile-"
                    "bootstrap 95% confidence interval"
                ),
                "binary_summary": "proportion with Wilson 95% confidence interval",
                "bootstrap_resamples": 10000,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "multiplicity": (
                    "no confirmatory null-hypothesis claims; sensitivity results "
                    "are labelled exploratory"
                ),
                "seed_labels": (
                    "identical identifiers are used for reproducibility only; "
                    "they do not define matched stochastic experimental units"
                ),
                "between_method_contrasts": (
                    "descriptive only unless an independently justified blocking "
                    "or common-random-number design is supplied"
                ),
            },
            "hardware_repetition": {
                "minimum_matched_job_pairs": 10,
                "minimum_distinct_calibration_dates": 3,
                "conditions": ["baseline", "noise_managed"],
                "analysis": (
                    "paired differences with calibration date retained as a block; "
                    "one pair is not sufficient for a causal claim"
                ),
            },
            "required_outputs": {
                "solver_trials": "results/post_contingency_v1.json",
                "admm_histories": "results/admm_post_contingency_v1.json",
                "hardware_packages": "results/hardware_YYYYMMDDTHHMMSSZ/",
                "generated_solver_table": (
                    "generated/post_contingency_results_table.tex"
                ),
                "generated_admm_table": "generated/post_contingency_admm_table.tex",
                "generated_validation_table": (
                    "generated/post_contingency_validation_table.tex"
                ),
                "generated_numbers": "generated/post_contingency_numbers.tex",
                "generated_figure": "figures/post_contingency_objective_gaps.png",
                "manuscript_destination": (
                    "a new Results subsection after Hardware evidence status; "
                    "abstract/conclusion may be updated only after validation"
                ),
            },
        },
    }
    unsigned = dict(protocol)
    protocol["protocol_sha256"] = _sha256_record(unsigned)
    return protocol


def validate_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unexpected study-protocol schema")
    recorded = protocol.get("protocol_sha256")
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    if recorded != _sha256_record(unsigned):
        raise ValueError("study-protocol hash mismatch")
    primary = protocol["prospective_primary_protocol"]
    if not primary["topology_precheck"]["spanning_tree_feasible"]:
        raise ValueError("predeclared contingency has no spanning-tree topology")
    order = primary["decision_set"]["variable_order"]
    indices = [record["branch_index"] for record in order]
    faulted = primary["contingency"]["forced_open_branch_index"]
    if faulted in indices or len(indices) != len(set(indices)):
        raise ValueError("prospective variable order is invalid")


def _tex_escape(value: str) -> str:
    return str(value).replace("_", r"\_")


def write_protocol_artifacts(protocol: dict) -> None:
    validate_protocol(protocol)
    PROTOCOL_PATH.parent.mkdir(exist_ok=True)
    PROTOCOL_TABLE_PATH.parent.mkdir(exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    primary = protocol["prospective_primary_protocol"]
    legacy = protocol["retained_legacy_protocol"]
    table = [
        "% Generated by study_protocol.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Scope and predeclared regeneration protocol. The prospective row is a design specification, not a numerical result.}",
        r"\label{tab:study-protocol}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{lXX}",
        r"\toprule",
        r"Item & Retained artifact & Prospective primary study \\",
        r"\midrule",
        (
            "Scenario & No forced outage (base case) & Hypothetical N$-1$ outage: "
            + _tex_escape(primary["contingency"]["forced_open_branch_name"])
            + r" (branch "
            + str(primary["contingency"]["forced_open_branch_index"])
            + r") \\"
        ),
        (
            r"Decision design & Prefix sizes $\{4,6,8,10,12\}$; omitted switches forced closed & Full remaining "
            + str(primary["decision_set"]["n_binary_variables"])
            + r"-switch decision set; no size sweep \\"
        ),
        (
            "Stochastic replication & Three QAOA/SA seeds; one retained QRAO run & "
            + str(len(primary["qaoa"]["seeds"]))
            + r" predeclared independent runs per stochastic method \\"
        ),
        (
            r"QAOA & $p\in\{1,2\}$, 1024 shots, COBYLA 50 iterations; initial points not archived & "
            r"$p\in\{1,2\}$, 4096 shots, COBYLA 100 iterations, tolerance $10^{-4}$; seeded initial points \\"
        ),
        (
            r"SA & 300 iterations, $T_0=10$, cooling 0.98 & 1000 iterations, $T_0=10$, cooling 0.995 \\"
        ),
        (
            r"Statistics & Descriptive only & Raw QUBO gaps $f_Q(\boldsymbol b)-f_Q^\star$; median/IQR; within-method percentile-bootstrap 95\% CI; Wilson 95\% CI for proportions \\"
        ),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ]
    PROTOCOL_TABLE_PATH.write_text("\n".join(table) + "\n")

    order_rows = []
    primary_indices = {
        row["branch_index"] for row in primary["decision_set"]["variable_order"]
    }
    faulted = primary["contingency"]["forced_open_branch_index"]
    for row in legacy["variable_order_full"]:
        if row["branch_index"] == faulted:
            status = "forced open"
            primary_position = "--"
        elif row["branch_index"] in primary_indices:
            status = "decision"
            primary_position = str(
                next(
                    item["position"]
                    for item in primary["decision_set"]["variable_order"]
                    if item["branch_index"] == row["branch_index"]
                )
            )
        else:
            status = "not used"
            primary_position = "--"
        order_rows.append(
            f"{row['position']} & {row['branch_index']} & "
            f"{_tex_escape(row['name'])} & {primary_position} & {status} \\\\"
        )
    order_table = [
        "% Generated by study_protocol.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Archived binary-variable ordering. Zero-based vector positions $\ell$ map to physical branch indices $\pi(\ell)$.}",
        r"\label{tab:switch-order}",
        r"\small",
        r"\begin{tabular}{rrlrl}",
        r"\toprule",
        r"Legacy position & Branch & Physical element & Prospective position & Prospective status \\",
        r"\midrule",
        *order_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    SWITCH_ORDER_TABLE_PATH.write_text("\n".join(order_table) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        protocol = json.loads(PROTOCOL_PATH.read_text())
        validate_protocol(protocol)
        if protocol != build_protocol():
            raise RuntimeError("study_protocol.json differs from current code/network")
        print("study protocol check passed")
        return
    protocol = build_protocol()
    write_protocol_artifacts(protocol)
    print(f"wrote {PROTOCOL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
