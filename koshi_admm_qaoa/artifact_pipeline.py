"""Create and verify the paper's frozen, machine-readable result artifact.

The paper must cite only ``results/benchmark.json``.  CSV files, figures,
LaTeX tables/macros, and ``RESULTS_SUMMARY.md`` are derived outputs and are
recorded by hash in ``results/artifact_manifest.json``.

The initial v2 artifact is an explicit migration of the aggregate v1 JSON that
shipped with the repository.  Information that v1 did not retain (per-seed
bitstrings, SA physics scores, raw ADMM histories, and raw hardware counts) is
marked unavailable rather than reconstructed or inferred as a numerical result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
GENERATED = ROOT / "generated"
ARTIFACT = RESULTS / "benchmark.json"
LEGACY = RESULTS / "benchmark_legacy.json"
MANIFEST = RESULTS / "artifact_manifest.json"
STUDY_PROTOCOL = RESULTS / "study_protocol.json"

METHOD_LABELS = {
    "QAOA p1 noiseless": r"QAOA $p{=}1$ noiseless",
    "QAOA p2 noiseless": r"QAOA $p{=}2$ noiseless",
    "QAOA p1 noisy": r"QAOA $p{=}1$ noisy",
    "QRAO 3v": "QRAO 3v",
    "SA": "SA",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _qubo_audit(n: int) -> dict:
    import numpy as np

    from network_data import scaled_network
    from qubo_builder import build_reconfig_qubo

    _, meta = build_reconfig_qubo(scaled_network(n), build_program=False)
    components = {}
    for name, stats in meta["component_stats"].items():
        components[name] = stats
    return {
        "n_binary_variables": int(meta["n_binary_variables"]),
        "standard_qaoa_qubits": int(meta["n_qubits"]),
        "n_qubits": int(meta["n_qubits"]),  # legacy-compatible alias
        "objective_convention": meta["notation_and_objective_convention"],
        "target_closed_switches": int(meta["K_target"]),
        "constant": float(meta["constant"]),
        "variable_order_branch_indices": [
            int(value) for value in meta["variable_order_branch_indices"]
        ],
        "linear_coefficients": np.asarray(meta["linear"], dtype=float).tolist(),
        "upper_triangular_quadratic_coefficients": np.triu(
            np.asarray(meta["Q"], dtype=float), 1
        ).tolist(),
        "offdiagonal_union_nonzero": int(meta["n_offdiag"]),
        "components": components,
        "topology_encoding": meta["topology_encoding"],
        "counting_note": (
            "Component counts overlap: a pair can receive coefficients from more "
            "than one penalty. The union count is the nonzero count after summation."
        ),
    }


def _radiality_audit(n: int) -> dict:
    """Generate an exact, dependency-light audit of the topology encoding."""
    import numpy as np

    from network_data import scaled_network
    from qubo_builder import build_reconfig_qubo
    from radiality import (
        fixed_forest_status,
        solve_exact_enumeration,
        solve_spanning_tree_milp,
        solve_uniform_cardinality_qubo,
        surrogate_radiality_audit,
    )

    network = scaled_network(n)
    _, actual = build_reconfig_qubo(network, build_program=False)
    _, easy = build_reconfig_qubo(
        network,
        lambda_cycle=0.0,
        lambda_iso=0.0,
        build_program=False,
    )
    fixed = fixed_forest_status(network)
    tree = solve_spanning_tree_milp(network)
    prefix = solve_uniform_cardinality_qubo(easy)
    easy_exact = solve_exact_enumeration(easy)
    actual_exact = solve_exact_enumeration(actual)
    surrogate = surrogate_radiality_audit(network, actual)
    offdiag = actual["Q"][np.triu_indices(actual["n_qubits"], 1)]
    unique = sorted({round(float(value), 12) for value in offdiag})
    return {
        "revision": "exact-radiality-v1",
        "fixed_subgraph": fixed,
        "hard_spanning_tree": {
            "feasible": bool(tree["success"]),
            "status": tree["status"],
            "time_s": float(tree["time_s"]),
            "target_closed_switches": tree.get("target_closed_switches"),
            "linear_impedance_proxy_objective": tree.get("objective"),
            "returned_bits": (
                tree["x"].astype(int).tolist() if tree.get("x") is not None else None
            ),
        },
        "actual_historical_qubo": {
            "uniform_cardinality_plus_linear": bool(len(unique) <= 1),
            "unique_offdiagonal_coefficients": unique,
            "exact_objective_recomputed_by_enumeration": actual_exact["objective"],
            "exact_enumeration_time_s": actual_exact["time_s"],
            "one_current_enumerated_minimizer_variable_order": (
                actual_exact["x"].astype(int).tolist()
                if actual_exact.get("x") is not None else None
            ),
            "all_current_enumerated_minimizers_variable_order": (
                actual_exact["all_minimizers_variable_order"]
            ),
            **surrogate,
        },
        "cardinality_only_baseline": {
            "solver": prefix["solver"],
            "objective": prefix["objective"],
            "time_s": prefix["time_s"],
            "hamming_weight": prefix["hamming_weight"],
            "exact_enumeration_objective": easy_exact["objective"],
            "agrees_with_enumeration": bool(
                abs(prefix["objective"] - easy_exact["objective"]) <= 1.0e-9
            ),
            "scope": (
                "exact only for the linear plus uniform-all-pairs cardinality "
                "variant with cycle and anti-islanding coefficients set to zero"
            ),
        },
    }


def migrate_legacy() -> dict:
    from study_protocol import build_protocol, write_protocol_artifacts

    study_protocol = build_protocol()
    write_protocol_artifacts(study_protocol)
    if not LEGACY.exists():
        raise FileNotFoundError(f"legacy input not found: {LEGACY}")
    legacy_rows = json.loads(LEGACY.read_text())
    sizes = sorted({int(row["n"]) for row in legacy_rows})
    instances = []
    for n in sizes:
        rows = [row for row in legacy_rows if int(row["n"]) == n]
        first = rows[0]
        radiality_audit = _radiality_audit(n)
        exact_loss = _finite(first.get("exact_true_loss"))
        exact_connected = exact_loss is not None
        methods = []
        for row in rows:
            method = row["method"]
            if method.startswith("QAOA") or method == "SA":
                optimizer_runs = 3
            else:
                optimizer_runs = 1
            loss = _finite(row.get("true_loss"))
            connected = row.get("connected")
            feasible = row.get("feasible")
            if method.startswith("QAOA"):
                scope = "last optimizer seed only (legacy benchmark behavior)"
            elif method == "QRAO 3v":
                scope = "single QRAO run"
            else:
                scope = "not recorded by the legacy benchmark"
            methods.append(
                {
                    "method": method,
                    "method_qubits": int(row["method_qubits"]),
                    "optimizer_runs": optimizer_runs,
                    "approximation_ratio_mean": float(row["approx_mean"]),
                    "approximation_ratio_std": float(row.get("approx_std", 0.0)),
                    "exact_optimum_hit_fraction": float(row["success"]),
                    "optimizer_run_exact_qubo_hit_fraction": float(row["success"]),
                    "hit_fraction_definition": (
                        "fraction of independent optimizer runs whose returned "
                        "objective reached the exact QUBO optimum"
                    ),
                    "time_mean_s": float(row["time_mean"]),
                    "representative_fixed_topology_soc": {
                        "model_revision": "legacy-continuous-model-v1",
                        "current_model_claim_eligible": False,
                        "loss_mw": loss,
                        "connected": connected,
                        "radial": None,
                        "feasible": feasible,
                        "scope": scope,
                    },
                }
            )
        instances.append(
            {
                "n": n,
                "qubo": _qubo_audit(n),
                "radiality_audit": radiality_audit,
                "exact_qubo": {
                    "objective": float(first["exact_fval"]),
                    "objective_symbol": "f_Q_star",
                    "time_s": float(first["exact_time"]),
                    "one_minimizer_bits_variable_order": None,
                    "returned_bits": None,  # legacy-compatible alias
                    "retained_solver_bitstring_status": (
                        "not archived in the legacy aggregate artifact"
                    ),
                    "current_enumeration": {
                        "scope": (
                            "reconstructed from the fully archived current QUBO "
                            "coefficients; not the historical solver-returned bitstring"
                        ),
                        "one_minimizer_bits_variable_order": (
                            radiality_audit["actual_historical_qubo"][
                                "one_current_enumerated_minimizer_variable_order"
                            ]
                        ),
                        "all_minimizers_bits_variable_order": (
                            radiality_audit["actual_historical_qubo"][
                                "all_current_enumerated_minimizers_variable_order"
                            ]
                        ),
                    },
                    "fixed_topology_soc": {
                        "model_revision": "legacy-continuous-model-v1",
                        "current_model_claim_eligible": False,
                        "loss_mw": exact_loss,
                        "connected": exact_connected,
                        "radial": None,
                        "feasible": exact_connected,
                        "connected_inferred_from_legacy_loss": True,
                    },
                },
                "best_connected_fixed_topology_soc": {
                    "model_revision": "legacy-continuous-model-v1",
                    "current_model_claim_eligible": False,
                    "loss_mw": _finite(first.get("gt_best_loss")),
                    "search_type": "legacy brute-force enumeration",
                },
                "methods": methods,
            }
        )

    artifact = {
        "schema_version": 2,
        "artifact_id": "koshi-benchmark-v2",
        "status": "frozen migration of legacy aggregate results",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source_commit_at_migration": _git_commit(),
            "working_tree_dirty_at_migration": _git_dirty(),
            "legacy_input": "results/benchmark_legacy.json",
            "legacy_input_sha256": _sha256(LEGACY),
            "migration_script": "artifact_pipeline.py",
            "audited_code_sha256": {
                name: _sha256(ROOT / name)
                for name in (
                    "network_data.py", "qubo_builder.py", "power_model.py",
                    "ac_validation.py", "radiality.py", "solvers.py", "benchmark.py",
                    "artifact_pipeline.py", "hardware_evidence.py",
                    "phase3_hardware.py", "study_protocol.py",
                    "statistics_report.py", "run_post_contingency_sensitivity.py",
                    "make_post_contingency_admm.py", "post_contingency_pipeline.py",
                    "environment_report.py", "../requirements-lock.txt",
                )
            },
            "important_limitation": (
                "The legacy file stored aggregate solver metrics and only one "
                "representative physics score. It did not store per-seed bitstrings."
            ),
        },
        "benchmark_configuration": {
            "sizes": sizes,
            "qaoa_and_sa_seeds": [42, 7, 123],
            "qaoa_depths": [1, 2],
            "shots": 1024,
            "optimizer_max_iterations": 50,
            "legacy_source": "benchmark.py defaults",
            "contingency": None,
            "interpretation": "legacy base case; not post-contingency",
        },
        "study_protocol": {
            "file": "results/study_protocol.json",
            "protocol_revision": study_protocol["protocol_revision"],
            "protocol_sha256": study_protocol["protocol_sha256"],
            "prospective_results_status": "not generated",
        },
        "notation_contract": study_protocol["notation_contract"],
        "continuous_model_evidence": {
            "current_code_revision": "continuous-model-v2",
            "numeric_results_model_revision": "legacy-continuous-model-v1",
            "status": (
                "stale: continuous-model-v2 changes shedding, thermal limits, "
                "transformer taps, slack balance, and AC validation; all SOC "
                "numbers require regeneration"
            ),
        },
        "radiality_model_evidence": {
            "current_code_revision": "exact-radiality-v1",
            "historical_qubo_revision": "heuristic-topology-qubo-v1",
            "status": (
                "current exact topology audit; historical quantum results retain "
                "the heuristic QUBO and are not hard-radiality results"
            ),
        },
        "hardware_model_evidence": {
            "current_code_revision": "hardware-evidence-v2",
            "numeric_result_status": "not generated",
            "required_output": (
                "results/hardware_YYYYMMDDTHHMMSSZ/hardware_results.json"
            ),
            "manuscript_table_output": "generated/hardware_evidence_tables.tex",
            "admission_rule": (
                "hardware_evidence.validate_evidence_package must pass before "
                "any hardware number is inserted into the manuscript"
            ),
        },
        "instances": instances,
        "excluded_evidence": {
            "admm": {
                "status": "excluded from frozen claims",
                "reason": "No machine-readable ADMM iteration histories were released.",
            },
            "hardware": {
                "status": "excluded from frozen claims",
                "reason": (
                    "No raw counts, backend/calibration metadata, transpiled circuit "
                    "artifact, or exported job results were released."
                ),
                "required_schema": "hardware-evidence-v2",
                "required_metrics": [
                    "modal sample and objective",
                    "best-energy sampled bitstring and objective",
                    "complete exact-QUBO minimizer set and f_Q_star",
                    "sampled-objective mean and standard error",
                    "shot-level probability assigned to the exact-QUBO minimizer set",
                    "same-instance topology, SOC, and nonlinear AC validation",
                ],
            },
            "qrao_ablation": {
                "status": "excluded from frozen claims",
                "reason": "No machine-readable ablation results were released.",
            },
            "qiskit_admm_optimizer": {
                "status": "excluded from benchmark comparison",
                "reason": "The repository contains code but no numerical result artifact.",
            },
        },
    }
    RESULTS.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    return artifact


def load_artifact() -> dict:
    artifact = json.loads(ARTIFACT.read_text())
    if artifact.get("schema_version") != 2:
        raise ValueError("results/benchmark.json is not the v2 frozen artifact")
    return artifact


def _rows(artifact):
    for instance in artifact["instances"]:
        for method in instance["methods"]:
            yield instance, method


def _optimizer_run_hit_fraction(method):
    if "optimizer_run_exact_qubo_hit_fraction" in method:
        return method["optimizer_run_exact_qubo_hit_fraction"]
    return method["exact_optimum_hit_fraction"]


def write_csv(artifact):
    fields = [
        "n", "method", "method_qubits", "optimizer_runs",
        "approximation_ratio_mean", "approximation_ratio_std",
        "optimizer_run_exact_qubo_hit_fraction", "time_mean_s", "f_Q_star",
        "exact_time_s", "representative_soc_loss_mw", "representative_connected",
        "representative_feasible", "representative_scope",
        "representative_soc_model_revision", "current_model_claim_eligible",
        "best_connected_soc_loss_mw", "offdiagonal_union_nonzero",
        "cardinality_nonzero", "cycle_nonzero", "anti_islanding_nonzero",
        "fixed_cycle_rank", "hard_spanning_tree_feasible",
        "cycle_false_positive_radial_states", "cycle_false_negative_states",
        "global_qubo_minimizers_that_are_radial", "hard_radiality_objective_gap",
    ]
    with (RESULTS / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for instance, method in _rows(artifact):
            soc = method["representative_fixed_topology_soc"]
            components = instance["qubo"]["components"]
            radiality = instance["radiality_audit"]
            historical = radiality["actual_historical_qubo"]
            writer.writerow(
                {
                    "n": instance["n"],
                    "method": method["method"],
                    "method_qubits": method["method_qubits"],
                    "optimizer_runs": method["optimizer_runs"],
                    "approximation_ratio_mean": method["approximation_ratio_mean"],
                    "approximation_ratio_std": method["approximation_ratio_std"],
                    "optimizer_run_exact_qubo_hit_fraction": (
                        _optimizer_run_hit_fraction(method)
                    ),
                    "time_mean_s": method["time_mean_s"],
                    "f_Q_star": instance["exact_qubo"]["objective"],
                    "exact_time_s": instance["exact_qubo"]["time_s"],
                    "representative_soc_loss_mw": soc["loss_mw"],
                    "representative_connected": soc["connected"],
                    "representative_feasible": soc["feasible"],
                    "representative_scope": soc["scope"],
                    "representative_soc_model_revision": soc.get("model_revision"),
                    "current_model_claim_eligible": soc.get(
                        "current_model_claim_eligible", True
                    ),
                    "best_connected_soc_loss_mw": instance[
                        "best_connected_fixed_topology_soc"
                    ]["loss_mw"],
                    "offdiagonal_union_nonzero": instance["qubo"][
                        "offdiagonal_union_nonzero"
                    ],
                    "cardinality_nonzero": components["cardinality"]["quadratic"][
                        "nonzero"
                    ],
                    "cycle_nonzero": components["cycle"]["quadratic"]["nonzero"],
                    "anti_islanding_nonzero": components["anti_islanding"][
                        "quadratic"
                    ]["nonzero"],
                    "fixed_cycle_rank": radiality["fixed_subgraph"]["fixed_cycle_rank"],
                    "hard_spanning_tree_feasible": radiality["hard_spanning_tree"][
                        "feasible"
                    ],
                    "cycle_false_positive_radial_states": historical[
                        "cycle_surrogate_false_positive_radial_states"
                    ],
                    "cycle_false_negative_states": historical[
                        "cycle_surrogate_false_negative_nonradial_target_cardinality_states"
                    ],
                    "global_qubo_minimizers_that_are_radial": historical[
                        "global_qubo_minimizers_that_are_radial"
                    ],
                    "hard_radiality_objective_gap": historical[
                        "hard_radiality_objective_gap"
                    ],
                }
            )


def _method(instance, name):
    return next(method for method in instance["methods"] if method["method"] == name)


def write_figures(artifact):
    FIGURES.mkdir(exist_ok=True)
    instances = artifact["instances"]
    sizes = [instance["n"] for instance in instances]
    methods = list(METHOD_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for method_name in methods:
        values = [_method(instance, method_name) for instance in instances]
        axes[0].errorbar(
            sizes,
            [value["approximation_ratio_mean"] for value in values],
            yerr=[value["approximation_ratio_std"] for value in values],
            marker="o",
            capsize=3,
            label=method_name,
        )
        axes[1].plot(
            sizes,
            [_optimizer_run_hit_fraction(value) for value in values],
            marker="o",
            label=method_name,
        )
    axes[0].axhline(1.0, ls="--", color="gray", lw=1)
    axes[0].set_ylabel("QUBO objective ratio (offset-sensitive)")
    axes[1].set_ylabel("optimizer-run exact-QUBO hit fraction")
    for axis in axes:
        axis.set_xlabel("number of switch variables")
        axis.set_xticks(sizes)
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Returned-objective ratio")
    axes[1].set_title("Independent runs attaining $f_Q^\\star$")
    fig.tight_layout()
    fig.savefig(FIGURES / "approx_vs_size.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method_name in methods:
        ax.plot(
            sizes,
            [_method(instance, method_name)["time_mean_s"] for instance in instances],
            marker="o",
            label=method_name,
        )
    ax.plot(
        sizes,
        [instance["exact_qubo"]["time_s"] for instance in instances],
        "k--",
        marker="x",
        label="classical exact QUBO",
    )
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel("number of switch variables")
    ax.set_ylabel("wall time (s, log scale)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "time_vs_size.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(sizes, sizes, "o-", label="QAOA")
    ax.plot(
        sizes,
        [_method(instance, "QRAO 3v")["method_qubits"] for instance in instances],
        "s-",
        label="QRAO 3v",
    )
    ax.set_xticks(sizes)
    ax.set_xlabel("number of switch variables")
    ax.set_ylabel("reported qubit count")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "qubits_vs_size.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        sizes,
        [instance["qubo"]["offdiagonal_union_nonzero"] for instance in instances],
        "o-",
        label="union after summation",
    )
    for component, label, style in [
        ("cardinality", "cardinality component", "--"),
        ("cycle", "cycle component", ":"),
        ("anti_islanding", "anti-islanding component", "-."),
    ]:
        ax.plot(
            sizes,
            [
                instance["qubo"]["components"][component]["quadratic"]["nonzero"]
                for instance in instances
            ],
            style,
            marker="o",
            label=label,
        )
    ax.set_xticks(sizes)
    ax.set_xlabel("number of switch variables")
    ax.set_ylabel("nonzero off-diagonal pairs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "qubo_coupling_growth.png", dpi=160)
    plt.close(fig)


def _tex_method(name):
    return METHOD_LABELS.get(name, name.replace("_", r"\_"))


def _tex_text(value):
    """Escape a plain-text provenance value for a LaTeX table cell."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _soc_cell(record):
    if record is None:
        return "NR"
    if record.get("connected") is False or record.get("feasible") is False:
        return "D-v1" if not record.get("current_model_claim_eligible", True) else "D"
    loss = record.get("loss_mw")
    if loss is None:
        return "NR"
    suffix = "v1 C" if not record.get("current_model_claim_eligible", True) else "C"
    return f"{loss:.2f} ({suffix})"


def write_tex(artifact):
    GENERATED.mkdir(exist_ok=True)
    instances = artifact["instances"]
    sizes = [instance["n"] for instance in instances]
    qaoa12 = _method(instances[-1], "QAOA p1 noiseless")
    exact_max_ms = 1000 * max(instance["exact_qubo"]["time_s"] for instance in instances)
    qaoa_times = [_method(instance, "QAOA p1 noiseless")["time_mean_s"] for instance in instances]
    noisy_max = max(_method(instance, "QAOA p1 noisy")["time_mean_s"] for instance in instances)
    macros = [
        "% Generated by artifact_pipeline.py; do not edit by hand.",
        rf"\newcommand{{\BenchmarkSizeSet}}{{\{{{','.join(map(str, sizes))}\}}}}",
        rf"\newcommand{{\MinQuboCouplings}}{{{instances[0]['qubo']['offdiagonal_union_nonzero']}}}",
        rf"\newcommand{{\MaxQuboCouplings}}{{{instances[-1]['qubo']['offdiagonal_union_nonzero']}}}",
        rf"\newcommand{{\QAOApOneTwelveRatio}}{{{qaoa12['approximation_ratio_mean']:.3f}}}",
        rf"\newcommand{{\QAOApOneTwelveHitFraction}}{{{_optimizer_run_hit_fraction(qaoa12):.2f}}}",
        rf"\newcommand{{\ExactQuboMaxTimeMs}}{{{exact_max_ms:.0f}}}",
        rf"\newcommand{{\QAOApOneMinTime}}{{{min(qaoa_times):.1f}}}",
        rf"\newcommand{{\QAOApOneMaxTime}}{{{max(qaoa_times):.1f}}}",
        rf"\newcommand{{\NoisyQaoaMaxTime}}{{{noisy_max:.0f}}}",
        r"\newcommand{\QraoReportedCompression}{1.00\times}",
    ]
    infeasible_sizes = [
        instance["n"] for instance in instances
        if not instance["radiality_audit"]["hard_spanning_tree"]["feasible"]
    ]
    prefix_max_ms = 1000 * max(
        instance["radiality_audit"]["cardinality_only_baseline"]["time_s"]
        for instance in instances
    )
    macros.extend(
        [
            rf"\newcommand{{\RadialInfeasibleSizeSet}}{{\{{{','.join(map(str, infeasible_sizes))}\}}}}",
            rf"\newcommand{{\PrefixBaselineMaxTimeMs}}{{{prefix_max_ms:.3f}}}",
        ]
    )
    (GENERATED / "paper_numbers.tex").write_text("\n".join(macros) + "\n")

    header = " & ".join(["$n$"] + [str(n) for n in sizes]) + r" \\"
    component_rows = []
    for key, label in [
        ("cardinality", "Cardinality component"),
        ("cycle", "Cycle component"),
        ("anti_islanding", "Anti-islanding component"),
    ]:
        values = [
            instance["qubo"]["components"][key]["quadratic"]["nonzero"]
            for instance in instances
        ]
        component_rows.append(label + " & " + " & ".join(map(str, values)) + r" \\")
    union = [instance["qubo"]["offdiagonal_union_nonzero"] for instance in instances]
    coupling = [
        "% Generated by artifact_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Audited nonzero off-diagonal QUBO pairs. Component counts can overlap; the union row counts nonzero pairs after all coefficients are summed.}",
        r"\label{tab:qubo-growth}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        header,
        r"\midrule",
        "Union after summation & " + " & ".join(map(str, union)) + r" \\",
        *component_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (GENERATED / "coupling_table.tex").write_text("\n".join(coupling) + "\n")

    radiality_rows = []
    for instance in instances:
        audit = instance["radiality_audit"]
        historical = audit["actual_historical_qubo"]
        feasible = "yes" if audit["hard_spanning_tree"]["feasible"] else "no"
        radial_minimum = (
            f"{historical['global_qubo_minimizers_that_are_radial']}/"
            f"{historical['global_qubo_minimizers']}"
        )
        gap = historical["hard_radiality_objective_gap"]
        gap_text = f"{gap:.3f}" if gap is not None else "--"
        radiality_rows.append(
            f"{instance['n']} & {audit['fixed_subgraph']['fixed_cycle_rank']} & "
            f"{feasible} & {historical['radial_configurations']} & "
            f"{historical['cycle_surrogate_false_positive_radial_states']} & "
            f"{historical['cycle_surrogate_false_negative_nonradial_target_cardinality_states']} & "
            f"{radial_minimum} & {gap_text} \\\\"
        )
    radiality_table = [
        "% Generated by artifact_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Exhaustive topology audit of the historical scaling instances. FP counts radial states penalized by the pairwise cycle term; FN counts nonradial target-cardinality states receiving zero cycle penalty. The minimizer column is the number of radial members of $\mathcal Z_Q^\star$ divided by $|\mathcal Z_Q^\star|$. The final column is the best hard-radial QUBO value minus $f_Q^\star$; a dash means no radial state exists.}",
        r"\label{tab:radiality-audit}",
        r"\small",
        r"\begin{tabular}{rccccccc}",
        r"\toprule",
        "$n$ & Fixed cycle rank & Tree feasible & Radial states & Cycle FP & Cycle FN & Radial/total $\\mathcal Z_Q^\\star$ & Hard-radial QUBO gap \\\\",
        r"\midrule",
        *radiality_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (GENERATED / "radiality_audit_table.tex").write_text(
        "\n".join(radiality_table) + "\n"
    )

    model_evidence = artifact.get("continuous_model_evidence", {})
    stale_soc = str(model_evidence.get("status", "")).startswith("stale")
    if stale_soc:
        soc_caption = (
            "Legacy fixed-topology SOC audit retained for discrepancy tracing. "
            "All v1 losses and status codes predate continuous-model-v2 and "
            "are not eligible as current physical results. C denotes a "
            "recorded connected state; D denotes a disconnected state or a "
            "non-finite legacy SOC record whose cause was not separately "
            "archived; NR denotes not recorded."
        )
    else:
        soc_caption = (
            "Fixed-topology SOC and nonlinear-validation audit from the current "
            "frozen artifact. C denotes connected with a finite recorded SOC "
            "loss; D denotes disconnected or infeasible; NR denotes not recorded."
        )
    table = [
        "% Generated by artifact_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{soc_caption}}}",
        r"\label{tab:legacy-soc-audit}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Method & " + " & ".join(f"$n={n}$" for n in sizes) + r" \\",
        r"\midrule",
    ]
    exact_cells = [
        _soc_cell(instance["exact_qubo"]["fixed_topology_soc"])
        for instance in instances
    ]
    table.append("One exact-QUBO minimizer & " + " & ".join(exact_cells) + r" \\")
    for method_name in METHOD_LABELS:
        cells = [
            _soc_cell(_method(instance, method_name)["representative_fixed_topology_soc"])
            for instance in instances
        ]
        table.append(_tex_method(method_name) + " & " + " & ".join(cells) + r" \\")
    best_cells = []
    for instance in instances:
        loss = instance["best_connected_fixed_topology_soc"]["loss_mw"]
        best_cells.append(f"{loss:.2f}" if loss is not None else "NR")
    table.extend(
        [
            r"\midrule",
            ("Best connected SOC found (v1)" if stale_soc else "Best connected SOC found")
            + " & " + " & ".join(best_cells) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    (GENERATED / "soc_validation_table.tex").write_text("\n".join(table) + "\n")

    from network_data import build_full_network

    network = build_full_network()
    provenance_rows = []
    for branch in network.branches:
        length = "--" if branch.length_km is None else f"{branch.length_km:g}"
        name = _tex_text(branch.name)
        source = _tex_text(branch.source_ref)
        status = _tex_text(branch.parameter_status)
        provenance_rows.append(
            f"{branch.idx} & {name} & {branch.physical_units} & {length} & "
            f"{branch.r_pu:.5f} & {branch.x_pu:.5f} & "
            f"{branch.rating_mva:.0f} & {branch.tap_ratio_pu:.3f} & "
            f"{status} & {source} \\\\"
        )
    provenance_table = [
        "% Generated by artifact_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Branch-level model provenance. Ratings, conductor parameters, several lengths, transformer short-circuit reactances, and operating taps are engineering assumptions unless the source column states otherwise. Parallel transformer units are represented by one equivalent rating and reactance.}",
        r"\label{tab:branch-provenance}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabularx}{\textwidth}{rXrrrrrrXX}",
        r"\toprule",
        r"$k$ & Element & Units & km & $r$ (pu) & $x$ (pu) & MVA & Tap & Status & Source/assumption \\",
        r"\midrule",
        *provenance_rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ]
    (GENERATED / "branch_provenance_table.tex").write_text(
        "\n".join(provenance_table) + "\n"
    )


def write_summary(artifact):
    instances = artifact["instances"]
    lines = [
        "# Frozen Results Summary",
        "",
        "This file is generated from `results/benchmark.json`. The JSON artifact is the source of truth.",
        "",
        "## Audit status",
        "",
        "- The legacy field `success` is now named **optimizer-run exact-QUBO hit fraction**; it is not a shot-level bitstring probability.",
        "- Infinite/disconnected SOC outcomes are stored as JSON `null` plus explicit connectivity/feasibility flags.",
        "- SA physics scores, per-seed bitstrings, ADMM histories, QRAO-ablation records, and raw hardware evidence were not present in the release and are not reconstructed.",
        "- The retained SOC numbers were produced by legacy continuous-model-v1 and are stale after the continuous-model-v2 correction; they remain only for discrepancy tracing.",
        "- The historical pairwise cycle and anti-islanding terms are heuristic. An exact single-commodity-flow/tree audit is stored for every size.",
        "- The n=4,6,8 scaling instances cannot be radial because their non-decision switches were forced closed into a cyclic mandatory subgraph.",
        "- Hardware results remain excluded. The v2 evidence contract separates the modal sample, best-energy sample, exact-QUBO minimizer set, sampled-objective mean, and shot-level exact-set probability, and requires same-n physical validation.",
        "- No branch contingency was passed in the retained benchmark. It is a legacy base-case artifact, not post-contingency evidence.",
        "- `results/study_protocol.json` predeclares a separate hypothetical N-1 regeneration study; no prospective result has been back-filled.",
        "",
        "## Benchmark rows",
        "",
        "| n | method | objective ratio | optimizer-run exact-QUBO hit fraction | representative SOC status |",
        "|---:|---|---:|---:|---|",
    ]
    for instance, method in _rows(artifact):
        soc = method["representative_fixed_topology_soc"]
        if soc["connected"] is False or soc["feasible"] is False:
            status = "legacy-v1 disconnected/infeasible"
        elif soc["loss_mw"] is None:
            status = "not recorded"
        else:
            status = f"{soc['loss_mw']:.3f} MW, legacy-v1 connected"
        lines.append(
            f"| {instance['n']} | {method['method']} | "
            f"{method['approximation_ratio_mean']:.3f} | "
            f"{_optimizer_run_hit_fraction(method):.2f} | {status} |"
        )
    (ROOT / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")


def write_manifest():
    outputs = [
        RESULTS / "benchmark.csv",
        ROOT / "RESULTS_SUMMARY.md",
        GENERATED / "paper_numbers.tex",
        GENERATED / "coupling_table.tex",
        GENERATED / "radiality_audit_table.tex",
        GENERATED / "study_protocol_table.tex",
        GENERATED / "switch_order_table.tex",
        GENERATED / "soc_validation_table.tex",
        GENERATED / "branch_provenance_table.tex",
        FIGURES / "approx_vs_size.png",
        FIGURES / "time_vs_size.png",
        FIGURES / "qubits_vs_size.png",
        FIGURES / "qubo_coupling_growth.png",
        STUDY_PROTOCOL,
        RESULTS / "environment.json",
        ROOT.parent / "requirements-lock.txt",
    ]
    manifest = {
        "schema_version": 1,
        "source": ARTIFACT.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(ARTIFACT),
        "derived_files": {
            Path(os.path.relpath(path, ROOT)).as_posix(): _sha256(path)
            for path in outputs
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def generate(artifact):
    from study_protocol import build_protocol, write_protocol_artifacts
    from environment_report import build_report

    write_protocol_artifacts(build_protocol())
    (RESULTS / "environment.json").write_text(
        json.dumps(build_report(), indent=2, allow_nan=False) + "\n"
    )
    write_csv(artifact)
    write_figures(artifact)
    write_tex(artifact)
    write_summary(artifact)
    write_manifest()


def check():
    from study_protocol import build_protocol, validate_protocol

    artifact = load_artifact()
    if not MANIFEST.exists():
        raise FileNotFoundError("artifact manifest is missing; run artifact_pipeline.py")
    manifest = json.loads(MANIFEST.read_text())
    errors = []
    if manifest["source_sha256"] != _sha256(ARTIFACT):
        errors.append("results/benchmark.json hash differs from the manifest")
    for relative, expected in manifest["derived_files"].items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing derived file: {relative}")
        elif _sha256(path) != expected:
            errors.append(f"derived file differs from manifest: {relative}")
    if "Infinity" in ARTIFACT.read_text() or "NaN" in ARTIFACT.read_text():
        errors.append("artifact contains a non-standard JSON numeric token")
    if not STUDY_PROTOCOL.exists():
        errors.append("study protocol is missing")
    else:
        protocol = json.loads(STUDY_PROTOCOL.read_text())
        try:
            validate_protocol(protocol)
            if protocol != build_protocol():
                errors.append("study protocol differs from current code/network")
        except ValueError as exc:
            errors.append(str(exc))
    for relative, expected in artifact.get("provenance", {}).get(
        "audited_code_sha256", {}
    ).items():
        path = ROOT / relative
        if not path.exists() or _sha256(path) != expected:
            errors.append(f"audited code differs from frozen artifact: {relative}")
    paper = ROOT.parent / "main_2.tex"
    if paper.exists():
        paper_text = paper.read_text()
        required = [
            r"\input{koshi_admm_qaoa/generated/paper_numbers.tex}",
            r"\input{koshi_admm_qaoa/generated/coupling_table.tex}",
            r"\input{koshi_admm_qaoa/generated/radiality_audit_table.tex}",
            r"\input{koshi_admm_qaoa/generated/study_protocol_table.tex}",
            r"\input{koshi_admm_qaoa/generated/switch_order_table.tex}",
            r"\input{koshi_admm_qaoa/generated/soc_validation_table.tex}",
            r"\input{koshi_admm_qaoa/generated/branch_provenance_table.tex}",
            r"\label{sec:notation}",
            r"\label{eq:qubo-ising}",
            r"\label{sec:hardware-contract}",
            r"hardware\_evidence.py",
        ]
        for marker in required:
            if marker not in paper_text:
                errors.append(f"paper is missing generated input: {marker}")
        for stale_claim in (
            "0.976", "5.858", "tab:qrao-ablation",
            r"\ref{tab:hardware}", r"\label{tab:hardware}",
            r"\label{tab:ac-loss}", r"\mathcal Z^*",
            "is then maximised", "impedance $z_k", "success-rate",
        ):
            if stale_claim in paper_text:
                errors.append(f"paper still contains an excluded legacy claim: {stale_claim}")
    if errors:
        raise RuntimeError("\n".join(errors))
    print("artifact check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate-legacy", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    artifact = migrate_legacy() if args.migrate_legacy else load_artifact()
    generate(artifact)
    print(f"generated paper artifacts from {ARTIFACT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
