"""Dependency-light hardware evidence analysis and validation.

This module deliberately does not import Qiskit.  It can therefore validate an
exported hardware package on a clean machine and prevents the terminology bug
identified in the technical review:

* the modal bitstring is the most frequently observed sample;
* the best-energy sampled bitstring minimizes the QUBO over observed samples;
* the exact-QUBO minimizer set contains every bitstring attaining ``f_Q_star``;
* the sampled-objective mean is the count-weighted mean QUBO value, with a
  standard error computed from the empirical energy distribution.

These quantities can coincide, but are never treated as synonyms.
"""
from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


HARDWARE_EVIDENCE_SCHEMA_VERSION = 2
HARDWARE_EVIDENCE_REVISION = "hardware-evidence-v2"
ENERGY_TOLERANCE = 1.0e-9


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_record(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_raw_bitstring(raw: str) -> str:
    bitstring = "".join(str(raw).split())
    if not bitstring or any(bit not in "01" for bit in bitstring):
        raise ValueError(f"invalid measured bitstring: {raw!r}")
    return bitstring


def qiskit_bitstring_to_bits(raw: str, n_variables: int) -> np.ndarray:
    """Convert Qiskit's displayed classical-bit order to variable order."""
    bitstring = normalize_raw_bitstring(raw)
    if len(bitstring) < n_variables:
        raise ValueError(
            f"bitstring {raw!r} has {len(bitstring)} bits; expected {n_variables}"
        )
    return np.fromiter(
        (int(bit) for bit in bitstring[::-1][:n_variables]),
        dtype=int,
        count=n_variables,
    )


def bits_to_variable_string(bits: Sequence[int]) -> str:
    values = np.asarray(bits, dtype=int)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("binary vector contains a value outside {0,1}")
    return "".join(str(int(value)) for value in values)


def qubo_energy(bits: Sequence[int], meta: Mapping) -> float:
    values = np.asarray(bits, dtype=float)
    linear = np.asarray(meta["linear"], dtype=float)
    quadratic = np.asarray(meta["Q"], dtype=float)
    if values.shape != linear.shape:
        raise ValueError(f"bits have shape {values.shape}; expected {linear.shape}")
    return float(
        meta.get("constant", 0.0)
        + linear @ values
        + values @ np.triu(quadratic, 1) @ values
    )


def enumerate_exact_optima(meta: Mapping, max_variables: int = 24) -> dict:
    """Return ``f_Q_star`` and every exact-QUBO minimizer bitstring."""
    n = int(meta.get("n_qubits", len(meta["linear"])))
    if n > max_variables:
        raise ValueError(
            f"exact enumeration is limited to {max_variables} variables, got {n}"
        )
    best = np.inf
    optima = []
    for values in product((0, 1), repeat=n):
        bits = np.fromiter(values, dtype=int, count=n)
        energy = qubo_energy(bits, meta)
        if energy < best - ENERGY_TOLERANCE:
            best = energy
            optima = [bits.tolist()]
        elif abs(energy - best) <= ENERGY_TOLERANCE:
            optima.append(bits.tolist())
    return {
        "objective": float(best),
        "bitstrings_variable_order": optima,
        "degeneracy": len(optima),
    }


def _normalize_counts(counts: Mapping[str, int]) -> dict[str, int]:
    normalized = {}
    for raw, count in counts.items():
        bitstring = normalize_raw_bitstring(raw)
        count = int(count)
        if count < 0:
            raise ValueError(f"negative count for {raw!r}")
        normalized[bitstring] = normalized.get(bitstring, 0) + count
    if not normalized or sum(normalized.values()) <= 0:
        raise ValueError("counts contain no shots")
    return normalized


def summarize_counts(
    counts: Mapping[str, int],
    meta: Mapping,
    exact_reference: Optional[Mapping] = None,
) -> dict:
    """Compute modal, best-sample, exact-set, and sampled-mean metrics."""
    normalized = _normalize_counts(counts)
    n = int(meta.get("n_qubits", len(meta["linear"])))
    exact = dict(exact_reference or enumerate_exact_optima(meta))
    exact_objective = float(exact["objective"])
    exact_strings = {
        bits_to_variable_string(bits)
        for bits in exact["bitstrings_variable_order"]
    }
    records = []
    shots = int(sum(normalized.values()))
    for raw, count in sorted(normalized.items()):
        bits = qiskit_bitstring_to_bits(raw, n)
        energy = qubo_energy(bits, meta)
        records.append(
            {
                "raw_bitstring": raw,
                "bits_variable_order": bits.tolist(),
                "variable_string": bits_to_variable_string(bits),
                "count": count,
                "probability": count / shots,
                "objective": energy,
                "objective_gap_from_exact": energy - exact_objective,
            }
        )

    modal_count = max(record["count"] for record in records)
    modal_records = [record for record in records if record["count"] == modal_count]
    selected_modal = min(modal_records, key=lambda record: record["raw_bitstring"])
    best_energy = min(record["objective"] for record in records)
    best_records = [
        record
        for record in records
        if abs(record["objective"] - best_energy) <= ENERGY_TOLERANCE
    ]
    selected_best = min(best_records, key=lambda record: record["raw_bitstring"])

    mean = sum(record["count"] * record["objective"] for record in records) / shots
    if shots > 1:
        sample_variance = sum(
            record["count"] * (record["objective"] - mean) ** 2
            for record in records
        ) / (shots - 1)
    else:
        sample_variance = 0.0
    standard_deviation = math.sqrt(max(sample_variance, 0.0))
    standard_error = standard_deviation / math.sqrt(shots)
    exact_count = sum(
        record["count"]
        for record in records
        if record["variable_string"] in exact_strings
    )

    energy_histogram = {}
    for record in records:
        key = f"{record['objective']:.12g}"
        energy_histogram[key] = energy_histogram.get(key, 0) + record["count"]

    return {
        "shots": shots,
        "support_size": len(records),
        "terminology": {
            "exact_objective": "f_Q_star",
            "exact_bitstrings": "Z_Q_star",
            "shot_level_exact_set_probability": "p_shot_star",
            "sampled_objective_mean": "mu_Q_hat",
        },
        "bit_order": {
            "raw_counts": "Qiskit classical display order",
            "returned_bits": "meta.switch_branches / variable_order order",
            "conversion": "reverse raw bitstring, then take n_variables bits",
        },
        "modal_sample": {
            "is_unique": len(modal_records) == 1,
            "tied_raw_bitstrings": [record["raw_bitstring"] for record in modal_records],
            "selected_raw_bitstring": selected_modal["raw_bitstring"],
            "bits_variable_order": selected_modal["bits_variable_order"],
            "count": selected_modal["count"],
            "probability": selected_modal["probability"],
            "objective": selected_modal["objective"],
            "objective_gap_from_exact": selected_modal["objective_gap_from_exact"],
            "is_exact_optimum": selected_modal["variable_string"] in exact_strings,
            "is_exact_qubo_minimizer": (
                selected_modal["variable_string"] in exact_strings
            ),
        },
        "best_energy_sample": {
            "is_unique": len(best_records) == 1,
            "tied_raw_bitstrings": [record["raw_bitstring"] for record in best_records],
            "selected_raw_bitstring": selected_best["raw_bitstring"],
            "bits_variable_order": selected_best["bits_variable_order"],
            "count": selected_best["count"],
            "probability": selected_best["probability"],
            "objective": selected_best["objective"],
            "objective_gap_from_exact": selected_best["objective_gap_from_exact"],
            "is_exact_optimum": selected_best["variable_string"] in exact_strings,
            "is_exact_qubo_minimizer": (
                selected_best["variable_string"] in exact_strings
            ),
        },
        "exact_optimum": {
            "objective": exact_objective,
            "bitstrings_variable_order": exact["bitstrings_variable_order"],
            "degeneracy": int(exact.get("degeneracy", len(exact_strings))),
            "observed_count": exact_count,
            "observed_probability": exact_count / shots,
        },
        "sampled_objective": {
            "expectation": mean,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "standard_error": standard_error,
            "expectation_gap_from_exact": mean - exact_objective,
        },
        "energy_histogram": [
            {
                "objective": float(energy),
                "count": count,
                "probability": count / shots,
            }
            for energy, count in sorted(
                ((float(key), count) for key, count in energy_histogram.items())
            )
        ],
        "bitstring_records": records,
    }


def build_problem_record(
    net,
    meta: Mapping,
    exact_reference: Mapping,
    faulted_branches: Optional[Sequence[int]] = None,
    scenario: Optional[Mapping] = None,
) -> dict:
    record = {
        "network_name": net.name,
        "n_bus": int(net.n_bus),
        "n_branch": int(net.n_branch),
        "n_switches": int(meta["n_qubits"]),  # compatibility alias
        "n_binary_variables": int(meta.get("n_binary_variables", meta["n_qubits"])),
        "switch_branches": [int(index) for index in meta["switch_branches"]],
        "variable_order": [
            {
                "position": int(position),
                "symbol": f"z_{position}",
                "physical_branch_index": int(index),
            }
            for position, index in enumerate(meta["switch_branches"])
        ],
        "objective_symbol": "f_Q",
        "objective_sense": "minimize",
        "binary_to_pauli_mapping": "z_ell = (1 - Z_ell) / 2",
        "constant": float(meta.get("constant", 0.0)),
        "linear": np.asarray(meta["linear"], dtype=float).tolist(),
        "Q_upper": np.triu(np.asarray(meta["Q"], dtype=float), 1).tolist(),
        "K_target": int(meta["K_target"]),
        "topology_encoding": meta.get("topology_encoding"),
        "faulted_branches": sorted(
            int(index) for index in (faulted_branches or [])
        ),
        "scenario": _jsonable(scenario),
        "exact_qubo": _jsonable(exact_reference),
    }
    record["problem_fingerprint_sha256"] = sha256_record(record)
    return record


def meta_from_problem_record(problem: Mapping) -> dict:
    return {
        "n_qubits": int(problem["n_switches"]),
        "n_binary_variables": int(
            problem.get("n_binary_variables", problem["n_switches"])
        ),
        "switch_branches": list(problem["switch_branches"]),
        "constant": float(problem.get("constant", 0.0)),
        "linear": np.asarray(problem["linear"], dtype=float),
        "Q": np.asarray(problem["Q_upper"], dtype=float),
    }


def _assert_close(actual, expected, label, tolerance=1.0e-9):
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def validate_evidence_package(evidence_dir: Path) -> dict:
    """Recompute all count-derived metrics and verify package file hashes."""
    evidence_dir = Path(evidence_dir)
    evidence_path = evidence_dir / "hardware_results.json"
    evidence = json.loads(evidence_path.read_text())
    errors = []
    if evidence.get("schema_version") != HARDWARE_EVIDENCE_SCHEMA_VERSION:
        errors.append("unsupported hardware evidence schema")
    if evidence.get("evidence_revision") != HARDWARE_EVIDENCE_REVISION:
        errors.append("unexpected hardware evidence revision")
    problem = evidence.get("problem", {})
    fingerprint = problem.get("problem_fingerprint_sha256")
    unsigned_problem = dict(problem)
    unsigned_problem.pop("problem_fingerprint_sha256", None)
    if fingerprint != sha256_record(unsigned_problem):
        errors.append("problem fingerprint does not match the problem record")
    meta = meta_from_problem_record(problem)
    exact = problem.get("exact_qubo", {})
    try:
        recomputed_exact = enumerate_exact_optima(meta)
        _assert_close(
            exact["objective"],
            recomputed_exact["objective"],
            "problem.exact_qubo.objective",
        )
        recorded_optima = {
            bits_to_variable_string(bits)
            for bits in exact["bitstrings_variable_order"]
        }
        recomputed_optima = {
            bits_to_variable_string(bits)
            for bits in recomputed_exact["bitstrings_variable_order"]
        }
        if recorded_optima != recomputed_optima:
            errors.append(
                "recorded exact-QUBO minimizer bitstrings are incomplete or incorrect"
            )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"exact QUBO reference is invalid: {exc}")

    circuit = evidence.get("execution", {}).get("transpiled_circuit", {})
    execution = evidence.get("execution", {})
    circuit_path = evidence_dir / str(circuit.get("file", ""))
    if not circuit_path.exists():
        errors.append("transpiled circuit file is missing")
    elif circuit.get("sha256") != sha256_file(circuit_path):
        errors.append("transpiled circuit hash mismatch")
    bound = execution.get("bound_ansatz", {})
    bound_path = evidence_dir / str(bound.get("file", ""))
    if not bound_path.exists():
        errors.append("bound ansatz file is missing")
    elif bound.get("sha256") != sha256_file(bound_path):
        errors.append("bound ansatz hash mismatch")
    for field in (
        "backend_configuration_file",
        "backend_properties_file",
        "backend_target_file",
    ):
        path = evidence_dir / str(execution.get(field, ""))
        if not path.exists():
            errors.append(f"backend snapshot is missing: {field}")

    runs = evidence.get("runs", {})
    if set(runs) != {"baseline", "noise_managed"}:
        errors.append("runs must contain exactly baseline and noise_managed conditions")
    for run_name, run in runs.items():
        condition = run.get("condition", {})
        expected_options = {
            "baseline": (False, False),
            "noise_managed": (True, True),
        }.get(run_name)
        if expected_options and (
            bool(condition.get("measurement_twirling")) != expected_options[0]
            or bool(condition.get("dynamical_decoupling")) != expected_options[1]
        ):
            errors.append(f"{run_name}: condition options do not match its label")
        if run.get("problem_fingerprint_sha256") != fingerprint:
            errors.append(f"{run_name}: problem fingerprint mismatch")
        if run.get("transpiled_circuit_sha256") != circuit.get("sha256"):
            errors.append(f"{run_name}: circuit hash mismatch")
        if not run.get("job_id"):
            errors.append(f"{run_name}: job ID is missing")
        if not run.get("backend_name"):
            errors.append(f"{run_name}: backend name is missing")
        calibration = str(run.get("calibration_timestamp_utc") or "")
        if not calibration or calibration.startswith("unavailable"):
            errors.append(f"{run_name}: calibration timestamp is missing")
        counts_path = evidence_dir / str(run.get("counts_file", ""))
        if not counts_path.exists():
            errors.append(f"{run_name}: counts file is missing")
            continue
        counts = json.loads(counts_path.read_text())
        if run.get("counts_sha256") != sha256_file(counts_path):
            errors.append(f"{run_name}: counts hash mismatch")
        job_path = evidence_dir / str(run.get("job_metadata_file", ""))
        if not job_path.exists():
            errors.append(f"{run_name}: job metadata file is missing")
        elif run.get("job_metadata_sha256") != sha256_file(job_path):
            errors.append(f"{run_name}: job metadata hash mismatch")
        required_physics = {
            "modal_sample_same_n",
            "best_energy_sample_same_n",
            "exact_optimum_same_n",
        }
        if set(run.get("same_instance_physics", {})) != required_physics:
            errors.append(f"{run_name}: same-instance physics records are incomplete")
        try:
            recomputed = summarize_counts(counts, meta, exact)
            if recomputed["shots"] != int(run.get("shots", -1)):
                errors.append(f"{run_name}: shot count mismatch")
            for section, field in (
                ("modal_sample", "objective"),
                ("best_energy_sample", "objective"),
                ("sampled_objective", "expectation"),
                ("sampled_objective", "standard_error"),
                ("exact_optimum", "observed_probability"),
            ):
                _assert_close(
                    run["summary"][section][field],
                    recomputed[section][field],
                    f"{run_name}.{section}.{field}",
                )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{run_name}: {exc}")

    manifest_path = evidence_dir / "evidence_manifest.json"
    if not manifest_path.exists():
        errors.append("evidence manifest is missing")
    else:
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest.get("files", {}).items():
            path = evidence_dir / relative
            if not path.exists():
                errors.append(f"manifest file is missing: {relative}")
            elif sha256_file(path) != expected:
                errors.append(f"manifest hash mismatch: {relative}")

    if errors:
        raise ValueError("\n".join(errors))
    return {
        "valid": True,
        "evidence_path": str(evidence_path),
        "problem_fingerprint_sha256": fingerprint,
        "runs": sorted(evidence.get("runs", {})),
    }


def _tex_escape(value: str) -> str:
    return str(value).replace("_", r"\_")


def _physics_status(record: Mapping) -> str:
    if not record:
        return "NR"
    flags = [
        "C" if record.get("connected") else "D",
        "R" if record.get("radial") else "M",
        "SOC" if record.get("fixed_topology_soc_validated") else "SOC--",
        "AC" if record.get("nonlinear_ac_validated") else "AC--",
    ]
    loss = record.get("loss_mw")
    if loss is not None:
        flags.append(f"{float(loss):.3f} MW")
    return ", ".join(flags)


def write_latex_tables(evidence_dir: Path, output_path: Path) -> Path:
    """Write publication tables only after strict package validation passes."""
    validate_evidence_package(evidence_dir)
    evidence_dir = Path(evidence_dir)
    evidence = json.loads((evidence_dir / "hardware_results.json").read_text())
    problem = evidence["problem"]
    execution = evidence["execution"]
    distribution_rows = []
    physics_rows = []
    for run_name in ("baseline", "noise_managed"):
        condition_label = {
            "baseline": "Baseline",
            "noise_managed": "Noise-managed",
        }[run_name]
        run = evidence["runs"][run_name]
        summary = run["summary"]
        sampled = summary["sampled_objective"]
        distribution_rows.append(
            f"{condition_label} & \\texttt{{{_tex_escape(run['job_id'])}}} & "
            f"{summary['modal_sample']['objective']:.4f} & "
            f"{summary['best_energy_sample']['objective']:.4f} & "
            f"${sampled['expectation']:.4f}\\pm{sampled['standard_error']:.4f}$ & "
            f"{summary['exact_optimum']['observed_probability']:.4f} \\\\"
        )
        for candidate, summary_key, physics_key in (
            ("Modal", "modal_sample", "modal_sample_same_n"),
            ("Best-energy sample", "best_energy_sample", "best_energy_sample_same_n"),
        ):
            item = summary[summary_key]
            physics_rows.append(
                f"{condition_label} & {candidate} & {item['objective']:.4f} & "
                f"{item['probability']:.4f} & "
                f"{_physics_status(run['same_instance_physics'][physics_key])} \\\\"
            )
    exact = problem["exact_qubo"]
    exact_physics = evidence["runs"]["baseline"]["same_instance_physics"][
        "exact_optimum_same_n"
    ]
    physics_rows.append(
        f"Reference & Exact-QUBO minimizer & {exact['objective']:.4f} & -- & "
        f"{_physics_status(exact_physics)} \\\\"
    )
    lines = [
        "% Generated by hardware_evidence.py after strict package validation.",
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Same-instance hardware sampling metrics for $n="
            + str(problem["n_switches"])
            + r"$ on \texttt{"
            + _tex_escape(execution["backend_name"])
            + r"}. The modal sample, best-energy sample, sampled-objective mean $\widehat\mu_Q$, and shot-level exact-set probability $\widehat p_{\rm shot}^\star$ are distinct quantities.}"
        ),
        r"\label{tab:hardware-distribution}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Condition & Job ID & Modal $f_Q$ & Best-sample $f_Q$ & $\widehat\mu_Q\pm\mathrm{SE}$ & $\widehat p_{\rm shot}^\star$ \\",
        r"\midrule",
        *distribution_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Same-$n$ topology and physical validation for the hardware candidates. C/D denotes connected/disconnected; R/M denotes radial/meshed; SOC and AC indicate validated checks; a double dash indicates failure.}",
        r"\label{tab:hardware-physics}",
        r"\small",
        r"\begin{tabular}{llccl}",
        r"\toprule",
        r"Condition & Candidate & $f_Q$ & Shot frequency & Same-instance validation \\",
        r"\midrule",
        *physics_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def write_evidence_manifest(evidence_dir: Path, exclude: Iterable[str] = ()) -> Path:
    evidence_dir = Path(evidence_dir)
    excluded = set(exclude) | {"evidence_manifest.json"}
    files = {
        path.name: sha256_file(path)
        for path in sorted(evidence_dir.iterdir())
        if path.is_file() and path.name not in excluded
    }
    manifest = {
        "schema_version": 1,
        "evidence_revision": HARDWARE_EVIDENCE_REVISION,
        "files": files,
    }
    path = evidence_dir / "evidence_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--write-tex", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_evidence_package(args.evidence_dir), indent=2))
    if args.write_tex:
        print(write_latex_tables(args.evidence_dir, args.write_tex))