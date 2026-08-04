"""Validate prospective artifacts and generate manuscript-ready outputs.

Nothing is generated from a partial run.  The primary solver artifact,
sensitivity artifact, and ADMM artifact must all be complete and share the
predeclared protocol hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from study_protocol import build_protocol, validate_protocol


ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "results" / "post_contingency_v1.json"
SENSITIVITY = ROOT / "results" / "post_contingency_sensitivity_v1.json"
ADMM = ROOT / "results" / "admm_post_contingency_v1.json"
TABLE = ROOT / "generated" / "post_contingency_results_table.tex"
ADMM_TABLE = ROOT / "generated" / "post_contingency_admm_table.tex"
VALIDATION_TABLE = ROOT / "generated" / "post_contingency_validation_table.tex"
SENSITIVITY_TABLE = (
    ROOT / "generated" / "post_contingency_sensitivity_table.tex"
)
NUMBERS = ROOT / "generated" / "post_contingency_numbers.tex"
FIGURE = ROOT / "figures" / "post_contingency_objective_gaps.png"
MANIFEST = ROOT / "results" / "post_contingency_manifest.json"
MANUSCRIPT = ROOT.parent / "main_2.tex"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    text = path.read_text()
    if any(token in text for token in ("NaN", "Infinity", "-Infinity")):
        raise ValueError(f"{path.name}: non-standard JSON numeric token")
    return json.loads(text)


def _validate_trial_seeds(trials, seeds, label):
    recorded = [int(trial["seed"]) for trial in trials]
    if sorted(recorded) != sorted(seeds) or len(recorded) != len(set(recorded)):
        raise ValueError(f"{label}: trial seeds differ from the protocol")
    for trial in trials:
        if "objective_gap" not in trial or "raw_candidate" not in trial:
            raise ValueError(f"{label}: trial lacks raw gap/topology evidence")
        projection = trial.get("projection", {})
        required_projection = {
            "required",
            "success",
            "opened_switches",
            "closed_switches",
            "projected_bits_variable_order",
            "projected_qubo_objective",
            "projected_objective_change",
            "projected_topology",
            "projected_fixed_topology_validation",
        }
        if not required_projection.issubset(projection):
            raise ValueError(f"{label}: trial lacks complete projection evidence")


def validate_inputs():
    protocol = build_protocol()
    validate_protocol(protocol)
    for path in (PRIMARY, SENSITIVITY, ADMM):
        if not path.exists():
            raise FileNotFoundError(f"required prospective artifact is missing: {path}")
    primary, sensitivity, admm = map(_load, (PRIMARY, SENSITIVITY, ADMM))
    expected_hash = protocol["protocol_sha256"]
    for name, artifact in (
        ("primary", primary),
        ("sensitivity", sensitivity),
        ("admm", admm),
    ):
        if artifact.get("protocol_sha256") != expected_hash:
            raise ValueError(f"{name}: protocol hash mismatch")
        provenance = artifact.get("provenance", {})
        if not provenance.get("source_commit") or not provenance.get(
            "package_versions"
        ):
            raise ValueError(f"{name}: source/package provenance is incomplete")
        convention = (
            artifact.get("qubo", {}).get("objective_convention")
            if name == "primary"
            else artifact.get("objective_convention")
        )
        if not convention or (
            convention.get("objective_sense") != "minimize"
            or convention.get("binary_to_pauli_mapping")
            != "z_ell = (1 - Z_ell) / 2"
        ):
            raise ValueError(f"{name}: objective/bit convention is incomplete")
    if not str(primary.get("status", "")).startswith("primary solver loop complete"):
        raise ValueError("primary solver artifact is incomplete")
    if sensitivity.get("status") != "complete":
        raise ValueError("sensitivity artifact is incomplete")
    if admm.get("status") != "complete":
        raise ValueError("ADMM artifact is incomplete")

    seeds = protocol["prospective_primary_protocol"]["qaoa"]["seeds"]
    if len(primary.get("methods", [])) != 4:
        raise ValueError("primary artifact must contain QAOA p1/p2, QRAO, and SA")
    exact = primary.get("exact_qubo", {})
    if (
        not exact.get("all_minimizers_bits_variable_order")
        or exact.get("minimizer_count")
        != len(exact["all_minimizers_bits_variable_order"])
    ):
        raise ValueError("primary artifact lacks the complete exact-QUBO minimizer set")
    for method in primary["methods"]:
        _validate_trial_seeds(method["trials"], seeds, method["method"])
    if len(sensitivity.get("variants", [])) != 11:
        raise ValueError("sensitivity artifact must contain 11 predeclared variants")
    for variant in sensitivity["variants"]:
        _validate_trial_seeds(
            variant["qaoa_p1"]["trials"], seeds, variant["name"]
        )
    if len(admm.get("exact_z_update_rho_sensitivity", [])) != 3:
        raise ValueError("ADMM artifact lacks the three rho sensitivity runs")
    if len(admm.get("qaoa_z_update_seed_runs", [])) != len(seeds):
        raise ValueError("ADMM artifact lacks the seed-matched QAOA runs")
    for name, runs in (
        ("exact ADMM", admm["exact_z_update_rho_sensitivity"]),
        ("QAOA ADMM", admm["qaoa_z_update_seed_runs"]),
    ):
        for run in runs:
            for field in (
                "z_raw",
                "raw_topology",
                "raw_qubo_objective",
                "repair",
                "projected_qubo_objective",
                "projected_objective_change",
                "validation",
                "termination_reason",
            ):
                if field not in run:
                    raise ValueError(f"{name}: run lacks {field}")
    return protocol, primary, sensitivity, admm


def _raw_topology_rate(trials, key):
    values = [
        bool(trial["raw_candidate"]["topology"].get(key, False))
        for trial in trials
    ]
    return sum(values) / len(values)


def _projection_rate(trials):
    return sum(bool(trial["projection"]["required"]) for trial in trials) / len(trials)


def _projected_validation_rate(trials, key):
    values = [
        bool(
            trial["projection"]["projected_fixed_topology_validation"].get(
                key, False
            )
        )
        for trial in trials
    ]
    return sum(values) / len(values)


def _median_projected_validated_loss(trials):
    losses = [
        trial["projection"]["projected_fixed_topology_validation"].get("loss_mw")
        for trial in trials
        if trial["projection"]["projected_fixed_topology_validation"].get(
            "nonlinear_ac_validated", False
        )
    ]
    finite = [float(value) for value in losses if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _projected_max_violations(trials):
    """Return maximum voltage-bound and thermal-utilization excesses."""
    voltage_violations = []
    thermal_violations = []
    for trial in trials:
        validation = trial["projection"]["projected_fixed_topology_validation"]
        nonlinear = validation.get("nonlinear_ac") or {}
        vmin = nonlinear.get("vmin_pu", validation.get("vmin_pu"))
        vmax = nonlinear.get("vmax_pu", validation.get("vmax_pu"))
        if vmin is not None and vmax is not None:
            voltage_violations.append(
                max(0.0, 0.90 - float(vmin), float(vmax) - 1.10)
            )
        diagnostics = validation.get("diagnostics") or {}
        utilizations = [
            nonlinear.get("max_apparent_power_utilization"),
            nonlinear.get("max_current_utilization"),
            diagnostics.get("max_apparent_power_utilization"),
            diagnostics.get("max_current_utilization"),
        ]
        finite = [
            float(value)
            for value in utilizations
            if value is not None and np.isfinite(value)
        ]
        if finite:
            thermal_violations.append(max(0.0, max(finite) - 1.0))
    return (
        max(voltage_violations, default=0.0),
        max(thermal_violations, default=0.0),
    )


def _interval_text(interval):
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _tex_escape(value):
    return str(value).replace("_", r"\_")


def _sensitivity_setting(variant):
    name = variant["name"]
    parameters = variant["qubo_parameters"]
    if name == "primary":
        return "Primary", "--"
    if name.startswith("lambda_card_"):
        return r"$\lambda_{\mathrm{card}}$", f"{parameters['lambda_card']:g}"
    if name.startswith("lambda_cycle_"):
        return r"$\lambda_{\mathrm{cyc}}$", f"{parameters['lambda_cycle']:g}"
    if name.startswith("lambda_iso_"):
        return r"$\lambda_{\mathrm{iso}}$", f"{parameters['lambda_iso']:g}"
    if name.startswith("loss_bias_"):
        return "Loss bias", f"{parameters['loss_bias']:g}"
    if name.startswith("qaoa_max_iterations_"):
        return "COBYLA max iterations", str(variant["qaoa_max_iterations"])
    raise ValueError(f"unrecognized sensitivity variant: {name}")


def write_outputs(protocol, primary, sensitivity, admm):
    TABLE.parent.mkdir(exist_ok=True)
    FIGURE.parent.mkdir(exist_ok=True)
    rows = []
    for method in primary["methods"]:
        stats = method["offset_invariant_statistics"]
        gap = stats["objective_gap"]
        hits = stats.get(
            "optimizer_run_exact_qubo_hit", stats["exact_optimum_hit"]
        )
        timing = stats["wall_time_s"]
        rows.append(
            f"{method['method']} & {stats['n_trials']} & {gap['median']:.3f} & "
            f"{_interval_text(gap['bootstrap_95_ci_median'])} & "
            f"{hits['proportion']:.3f} & {_interval_text(hits['wilson_95_ci'])} & "
            f"{timing['median']:.3f} & "
            f"{_raw_topology_rate(method['trials'], 'radial'):.3f} & "
            f"{_projected_validation_rate(method['trials'], 'nonlinear_ac_validated'):.3f} \\\\"
        )
    table = [
        "% Generated by post_contingency_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Predeclared hypothetical N$-1$ post-contingency results. $\Delta_Q=f_Q(\boldsymbol{b})-f_Q^\star$. Gap intervals are within-method percentile-bootstrap 95\% intervals for the median; optimizer-run exact-QUBO hit intervals are Wilson 95\% intervals. Raw radiality is evaluated before projection; nonlinear AC validation is evaluated after the declared spanning-tree projection.}",
        r"\label{tab:post-contingency-results}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & $R$ & Median $\Delta_Q$ & 95\% CI & $\widehat h_{\rm run}^\star$ & 95\% CI & Median s & Raw radial & Projected nonlinear AC validated \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    TABLE.write_text("\n".join(table) + "\n")

    sensitivity_rows = []
    for variant in sensitivity["variants"]:
        factor, setting = _sensitivity_setting(variant)
        stats = variant["qaoa_p1"]["offset_invariant_statistics"]
        gap = stats["objective_gap"]
        hits = stats["optimizer_run_exact_qubo_hit"]
        sensitivity_rows.append(
            f"{factor} & {setting} & {stats['n_trials']} & "
            f"{gap['median']:.3f} & "
            f"{_interval_text(gap['bootstrap_95_ci_median'])} & "
            f"{hits['proportion']:.3f} & "
            f"{_interval_text(hits['wilson_95_ci'])} \\\\"
        )
    sensitivity_table = [
        "% Generated by post_contingency_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Predeclared one-factor-at-a-time QAOA-$p=1$ sensitivity audit. Each row contains 30 independent optimizer runs. Gap and hit statistics are defined relative to that row's own QUBO optimum; rows with different QUBO weights therefore diagnose tuning sensitivity but do not form a cross-objective performance ranking.}",
        r"\label{tab:post-contingency-sensitivity}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Varied factor & Setting & $R$ & Median $\Delta_Q$ & 95\% CI & $\widehat h_{\rm run}^\star$ & 95\% CI \\",
        r"\midrule",
        *sensitivity_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    SENSITIVITY_TABLE.write_text("\n".join(sensitivity_table) + "\n")

    labels = [
        method["method"].replace(" noiseless", "")
        for method in primary["methods"]
    ]
    gaps = [
        [trial["objective_gap"] for trial in method["trials"]]
        for method in primary["methods"]
    ]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.boxplot(gaps, tick_labels=labels, showmeans=True)
    ax.axhline(0.0, color="black", lw=1, ls="--")
    ax.set_ylabel(
        r"QUBO gap $\Delta_Q=f_Q(\boldsymbol{b})-f_Q^\star$",
        fontsize=9,
    )
    ax.set_title("Post-contingency seed distribution", fontsize=10)
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)

    exact_rows = []
    for result in admm["exact_z_update_rho_sensitivity"]:
        history = result["history"]
        exact_rows.append(
            f"Exact-QUBO $\\boldsymbol{{z}}$-update, $\\rho={result['configuration']['rho']}$ & 1 & "
            f"{result['iters']} & {history['primal'][-1]:.3e} & "
            f"{history['dual'][-1]:.3e} & "
            f"{_tex_escape(result['termination_reason'])} \\\\"
        )
    qaoa_runs = admm["qaoa_z_update_seed_runs"]
    final_primal = np.asarray([run["history"]["primal"][-1] for run in qaoa_runs])
    final_dual = np.asarray([run["history"]["dual"][-1] for run in qaoa_runs])
    converged = sum(
        run["termination_reason"] == "primal_and_dual_tolerances"
        for run in qaoa_runs
    )
    exact_rows.append(
        f"QAOA $\\boldsymbol{{z}}$-update, $\\rho=3$ & {len(qaoa_runs)} & -- & "
        f"{np.median(final_primal):.3e} & {np.median(final_dual):.3e} & "
        f"{converged}/{len(qaoa_runs)} converged \\\\"
    )
    admm_table = [
        "% Generated by post_contingency_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Predeclared ADMM stopping audit. QAOA residuals are medians over independent optimizer runs.}",
        r"\label{tab:post-contingency-admm}",
        r"\small",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Configuration & Runs & Iterations & Final primal & Final dual & Termination \\",
        r"\midrule",
        *exact_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    ADMM_TABLE.write_text("\n".join(admm_table) + "\n")

    validation_rows = []
    for method in primary["methods"]:
        trials = method["trials"]
        loss = _median_projected_validated_loss(trials)
        loss_text = "--" if loss is None else f"{loss:.3f}"
        cone_values = [
            (
                trial["projection"]["projected_fixed_topology_validation"].get(
                    "diagnostics"
                )
                or {}
            ).get("max_cone_slack_pu2")
            for trial in trials
        ]
        angle_values = [
            (
                (
                    trial["projection"]["projected_fixed_topology_validation"].get(
                        "diagnostics"
                    )
                    or {}
                ).get("angle_recovery")
                or {}
            ).get("max_residual_rad")
            for trial in trials
        ]
        cone = [float(value) for value in cone_values if value is not None]
        angle = [float(value) for value in angle_values if value is not None]
        max_cone = "--" if not cone else f"{max(abs(value) for value in cone):.2e}"
        max_angle = "--" if not angle else f"{max(abs(value) for value in angle):.2e}"
        max_voltage_violation, max_thermal_violation = _projected_max_violations(
            trials
        )
        validation_rows.append(
            f"{method['method']} & {len(trials)} & "
            f"{_raw_topology_rate(trials, 'connected'):.3f} & "
            f"{_raw_topology_rate(trials, 'radial'):.3f} & "
            f"{_projection_rate(trials):.3f} & "
            f"{_projected_validation_rate(trials, 'fixed_topology_soc_validated'):.3f} & "
            f"{_projected_validation_rate(trials, 'nonlinear_ac_validated'):.3f} & "
            f"{max_cone} & {max_angle} & "
            f"{max_voltage_violation:.2e} & {max_thermal_violation:.2e} & "
            f"{loss_text} \\\\"
        )
    validation_table = [
        "% Generated by post_contingency_pipeline.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Topology and physical-validation audit for the prospective solver trials. Raw rates use solver outputs before repair. Projection and validation rates use the explicitly archived projected candidates. Voltage violation is the maximum excess beyond $[0.90,1.10]$ pu; thermal violation is the maximum utilization excess above 1. Loss is the median only among projected candidates passing the series-only nonlinear AC validation; a dash means no validated loss.}",
        r"\label{tab:post-contingency-validation}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        r"Method & $R$ & Raw C & Raw R & Projected & SOC recovered & AC validated & Max cone slack & Max angle residual & Max $|V|$ viol. & Max thermal viol. & Validated loss MW \\",
        r"\midrule",
        *validation_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    VALIDATION_TABLE.write_text("\n".join(validation_table) + "\n")

    ranked_gap = sorted(
        primary["methods"],
        key=lambda method: (
            method["offset_invariant_statistics"]["objective_gap"]["median"],
            method["method"],
        ),
    )
    ranked_hits = sorted(
        primary["methods"],
        key=lambda method: (
            -method["offset_invariant_statistics"][
                "optimizer_run_exact_qubo_hit"
            ]["proportion"],
            method["method"],
        ),
    )
    best_gap = ranked_gap[0]
    best_hit = ranked_hits[0]
    best_gap_trials = best_gap["trials"]
    best_gap_stats = best_gap["offset_invariant_statistics"]["objective_gap"]
    best_gap_interval = best_gap_stats["bootstrap_95_ci_median"]
    sensitivity_medians = [
        variant["qaoa_p1"]["offset_invariant_statistics"]["objective_gap"][
            "median"
        ]
        for variant in sensitivity["variants"]
    ]
    numbers = [
        "% Generated by post_contingency_pipeline.py; do not edit by hand.",
        rf"\newcommand{{\PostContingencyRunCount}}{{{len(best_gap['trials'])}}}",
        rf"\newcommand{{\PostContingencyBestGapMethod}}{{{_tex_escape(best_gap['method'])}}}",
        rf"\newcommand{{\PostContingencyBestMedianGap}}{{{best_gap_stats['median']:.3f}}}",
        rf"\newcommand{{\PostContingencyBestMedianGapCILower}}{{{best_gap_interval[0]:.3f}}}",
        rf"\newcommand{{\PostContingencyBestMedianGapCIUpper}}{{{best_gap_interval[1]:.3f}}}",
        rf"\newcommand{{\PostContingencyBestHitMethod}}{{{_tex_escape(best_hit['method'])}}}",
        rf"\newcommand{{\PostContingencyBestHitFraction}}{{{best_hit['offset_invariant_statistics']['optimizer_run_exact_qubo_hit']['proportion']:.3f}}}",
        rf"\newcommand{{\PostContingencyBestGapRawConnectedRate}}{{{_raw_topology_rate(best_gap_trials, 'connected'):.3f}}}",
        rf"\newcommand{{\PostContingencyBestGapRawConnectedPercent}}{{{100 * _raw_topology_rate(best_gap_trials, 'connected'):.1f}}}",
        rf"\newcommand{{\PostContingencyBestGapRawRadialRate}}{{{_raw_topology_rate(best_gap_trials, 'radial'):.3f}}}",
        rf"\newcommand{{\PostContingencyBestGapRawRadialPercent}}{{{100 * _raw_topology_rate(best_gap_trials, 'radial'):.1f}}}",
        rf"\newcommand{{\PostContingencyBestGapProjectionRate}}{{{_projection_rate(best_gap_trials):.3f}}}",
        rf"\newcommand{{\PostContingencyBestGapProjectionPercent}}{{{100 * _projection_rate(best_gap_trials):.1f}}}",
        rf"\newcommand{{\PostContingencyBestGapSocRate}}{{{_projected_validation_rate(best_gap_trials, 'fixed_topology_soc_validated'):.3f}}}",
        rf"\newcommand{{\PostContingencyBestGapSocPercent}}{{{100 * _projected_validation_rate(best_gap_trials, 'fixed_topology_soc_validated'):.1f}}}",
        rf"\newcommand{{\PostContingencyBestGapAcRate}}{{{_projected_validation_rate(best_gap_trials, 'nonlinear_ac_validated'):.3f}}}",
        rf"\newcommand{{\PostContingencyBestGapAcPercent}}{{{100 * _projected_validation_rate(best_gap_trials, 'nonlinear_ac_validated'):.1f}}}",
        rf"\newcommand{{\PostContingencyExactOptimumCount}}{{{primary['exact_qubo']['minimizer_count']}}}",
        rf"\newcommand{{\PostContingencySensitivityVariantCount}}{{{len(sensitivity_medians)}}}",
        rf"\newcommand{{\PostContingencySensitivityMinMedianGap}}{{{min(sensitivity_medians):.3f}}}",
        rf"\newcommand{{\PostContingencySensitivityMaxMedianGap}}{{{max(sensitivity_medians):.3f}}}",
    ]
    NUMBERS.write_text("\n".join(numbers) + "\n")

    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "sources": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (PRIMARY, SENSITIVITY, ADMM)
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                TABLE,
                ADMM_TABLE,
                VALIDATION_TABLE,
                SENSITIVITY_TABLE,
                NUMBERS,
                FIGURE,
            )
        },
        "sensitivity_variants": len(sensitivity["variants"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def check_outputs():
    protocol, primary, sensitivity, admm = validate_inputs()
    if not MANIFEST.exists():
        raise FileNotFoundError("post-contingency manifest is missing")
    manifest = _load(MANIFEST)
    if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("post-contingency manifest protocol hash mismatch")
    for group in ("sources", "outputs"):
        for relative, expected in manifest[group].items():
            path = ROOT / relative
            if not path.exists() or _sha256(path) != expected:
                raise ValueError(f"post-contingency {group} hash mismatch: {relative}")
    manuscript = MANUSCRIPT.read_text()
    required_inputs = (
        r"\input{koshi_admm_qaoa/generated/post_contingency_numbers.tex}",
        r"\input{koshi_admm_qaoa/generated/post_contingency_results_table.tex}",
        r"\input{koshi_admm_qaoa/generated/post_contingency_validation_table.tex}",
        r"\input{koshi_admm_qaoa/generated/post_contingency_sensitivity_table.tex}",
        r"\input{koshi_admm_qaoa/generated/post_contingency_admm_table.tex}",
        "koshi_admm_qaoa/figures/post_contingency_objective_gaps.png",
    )
    missing = [token for token in required_inputs if token not in manuscript]
    if missing:
        raise ValueError(
            "manuscript lacks prospective generated inputs: "
            + ", ".join(missing)
        )
    for stale in (
        "it has not been executed",
        "neither file exists in this revision",
        "numerical benchmark has not yet been regenerated",
        "It is a prospective design,\nnot a result",
    ):
        if stale in manuscript:
            raise ValueError(f"manuscript retains stale prospective status: {stale}")
    if manuscript.count(r"\PostContingencyBestMedianGap") < 2:
        raise ValueError(
            "generated prospective headline macro is not used in both "
            "headline prose locations"
        )
    return protocol, primary, sensitivity, admm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_outputs()
        print("post-contingency artifact check passed")
        return
    inputs = validate_inputs()
    write_outputs(*inputs)
    print(f"generated {TABLE.relative_to(ROOT)}, {ADMM_TABLE.relative_to(ROOT)}, and {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
