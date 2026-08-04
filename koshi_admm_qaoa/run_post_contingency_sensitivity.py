"""Run the predeclared QUBO-weight and QAOA-budget sensitivity study."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from benchmark import (
    _git_commit,
    _jsonable,
    _method_record,
    _soc_record,
    _trial,
    _versions,
)
from network_data import build_full_network
from qubo_builder import build_reconfig_qubo
import solvers
from study_protocol import build_protocol, validate_protocol


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "post_contingency_sensitivity_v1.json"


def _sensitivity_trial(task):
    variant, primary, seed, fopt = task
    faulted = [primary["contingency"]["forced_open_branch_index"]]
    network = build_full_network()
    qp, meta = build_reconfig_qubo(
        network, faulted=faulted, **variant["qubo_parameters"]
    )
    result = solvers.solve_qaoa(
        qp,
        reps=1,
        kind="noiseless",
        shots=primary["qaoa"]["shots"],
        maxiter=variant["qaoa_max_iterations"],
        seed=seed,
        optimizer_tol=primary["qaoa"]["optimizer_tolerance"],
    )
    score = solvers.score_config(network, result["x"], meta, faulted=faulted)
    return _trial(
        seed, result, score, fopt, network, meta, faulted=faulted
    )


def _save(artifact):
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(
        json.dumps(_jsonable(artifact), indent=2, allow_nan=False) + "\n"
    )


def _variants(primary):
    base = {
        key: primary["qubo_parameters"][key]
        for key in ("lambda_card", "lambda_cycle", "lambda_iso", "loss_bias")
    }
    yield {
        "name": "primary",
        "qubo_parameters": dict(base),
        "qaoa_max_iterations": primary["qaoa"]["optimizer_max_iterations"],
    }
    for parameter in base:
        for multiplier in (0.5, 2.0):
            weights = dict(base)
            weights[parameter] *= multiplier
            yield {
                "name": f"{parameter}_x{multiplier:g}",
                "qubo_parameters": weights,
                "qaoa_max_iterations": primary["qaoa"]["optimizer_max_iterations"],
            }
    for budget in (50, 200):
        yield {
            "name": f"qaoa_max_iterations_{budget}",
            "qubo_parameters": dict(base),
            "qaoa_max_iterations": budget,
        }


def run():
    protocol = build_protocol()
    validate_protocol(protocol)
    primary = protocol["prospective_primary_protocol"]
    checkpoint = None
    if OUTPUT.exists():
        try:
            candidate = json.loads(OUTPUT.read_text())
            if candidate.get("protocol_sha256") == protocol["protocol_sha256"]:
                checkpoint = candidate
        except (OSError, ValueError, TypeError):
            checkpoint = None
    faulted = [primary["contingency"]["forced_open_branch_index"]]
    network = build_full_network()
    run_commit = _git_commit()
    artifact = {
        "schema_version": 1,
        "artifact_id": "koshi-post-contingency-sensitivity-v1",
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_revision": protocol["protocol_revision"],
        "protocol_sha256": protocol["protocol_sha256"],
        "provenance": {
            "source_commit": run_commit,
            "package_versions": _versions(),
            "generator": "run_post_contingency_sensitivity.run",
        },
        "objective_convention": {
            "objective_symbol": "f_Q",
            "objective_sense": "minimize",
            "binary_to_pauli_mapping": "z_ell = (1 - Z_ell) / 2",
            "returned_array_order": "QuadraticProgram variable order",
        },
        "scenario": primary["contingency"],
        "design": primary["sensitivity_analysis"],
        "variants": [],
    }
    expected_variants = {variant["name"]: variant for variant in _variants(primary)}
    if checkpoint is not None:
        checkpoint_commit = checkpoint.get("provenance", {}).get("source_commit")
        for variant in checkpoint.get("variants", []):
            name = variant.get("name")
            trials = variant.get("qaoa_p1", {}).get("trials", [])
            seeds = [trial.get("seed") for trial in trials]
            expected = expected_variants.get(name)
            if (
                expected is not None
                and variant.get("qubo_parameters") == expected["qubo_parameters"]
                and variant.get("qaoa_max_iterations")
                == expected["qaoa_max_iterations"]
                and len(seeds) == len(primary["qaoa"]["seeds"])
                and sorted(seeds) == sorted(primary["qaoa"]["seeds"])
                and len(seeds) == len(set(seeds))
            ):
                variant["source_commit"] = variant.get(
                    "source_commit", checkpoint_commit
                )
                artifact["variants"].append(variant)
        if artifact["variants"]:
            artifact["provenance"]["resumed_from_source_commit"] = checkpoint_commit
            artifact["provenance"]["reused_variants"] = [
                variant["name"] for variant in artifact["variants"]
            ]
        partial = checkpoint.get("in_progress_variant", {})
        expected = expected_variants.get(partial.get("name"))
        partial_seeds = [
            trial.get("seed") for trial in partial.get("trials", [])
        ]
        if (
            expected is not None
            and partial.get("qubo_parameters") == expected["qubo_parameters"]
            and partial.get("qaoa_max_iterations")
            == expected["qaoa_max_iterations"]
            and set(partial_seeds).issubset(set(primary["qaoa"]["seeds"]))
            and len(partial_seeds) == len(set(partial_seeds))
        ):
            artifact["in_progress_variant"] = partial
    _save(artifact)

    for variant in _variants(primary):
        if any(
            completed["name"] == variant["name"]
            for completed in artifact["variants"]
        ):
            continue
        qp, meta = build_reconfig_qubo(
            network, faulted=faulted, **variant["qubo_parameters"]
        )
        exact = solvers.solve_exact_qubo(qp)
        fopt = exact["fval"]
        exact_score = solvers.score_config(
            network, exact["x"], meta, faulted=faulted
        )
        partial = artifact.get("in_progress_variant", {})
        if partial.get("name") == variant["name"]:
            trials = list(partial.get("trials", []))
            source_commits = set(partial.get("source_commits", []))
        else:
            trials = []
            source_commits = set()
        completed_seeds = {trial["seed"] for trial in trials}
        tasks = [
            (variant, primary, seed, fopt)
            for seed in primary["qaoa"]["seeds"]
            if seed not in completed_seeds
        ]
        with ProcessPoolExecutor(
            max_workers=primary["execution"]["parallel_workers"],
            max_tasks_per_child=1,
        ) as pool:
            for trial in pool.map(_sensitivity_trial, tasks):
                trials.append(trial)
                source_commits.add(run_commit)
                artifact["in_progress_variant"] = {
                    **variant,
                    "trials": trials,
                    "source_commits": sorted(source_commits),
                }
                _save(artifact)
        trials.sort(
            key=lambda trial: primary["qaoa"]["seeds"].index(trial["seed"])
        )
        artifact["variants"].append(
            {
                **variant,
                "source_commit": run_commit,
                "source_commits": sorted(source_commits),
                "exact_qubo": {
                    "objective": float(fopt),
                    "returned_bits": _jsonable(exact["x"]),
                    "time_s": float(exact["time_s"]),
                    "fixed_topology_validation": _soc_record(exact_score),
                },
                "qaoa_p1": _method_record(
                    "QAOA p1 noiseless", meta["n_qubits"], trials
                ),
            }
        )
        artifact.pop("in_progress_variant", None)
        _save(artifact)

    artifact["status"] = "complete"
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _save(artifact)
    return artifact


if __name__ == "__main__":
    run()
    print(f"Saved {OUTPUT}")
