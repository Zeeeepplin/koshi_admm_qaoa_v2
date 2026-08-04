"""Tests for the predeclared statistical summaries."""
from __future__ import annotations

import numpy as np

from post_contingency_pipeline import _projected_max_violations
from statistics_report import (
    bootstrap_median_interval,
    summarize_paired_gap_difference,
    summarize_trials,
    wilson_interval,
)


def test_wilson_interval_contains_observed_proportion():
    lower, upper = wilson_interval(15, 30)
    assert lower < 0.5 < upper
    assert 0 <= lower <= upper <= 1


def test_bootstrap_interval_is_reproducible():
    values = [0, 1, 2, 3, 4, 5]
    first = bootstrap_median_interval(values, resamples=1000, seed=7)
    second = bootstrap_median_interval(values, resamples=1000, seed=7)
    assert first == second
    assert first[0] <= np.median(values) <= first[1]


def test_trial_and_paired_summaries_use_raw_gaps():
    left = [
        {"seed": seed, "objective_gap": gap, "time_s": 1 + seed / 10, "reached_exact_qubo_optimum": gap == 0}
        for seed, gap in zip((1, 2, 3, 4), (0.0, 0.2, 0.1, 0.3))
    ]
    right = [
        {"seed": seed, "objective_gap": gap, "time_s": 2 + seed / 10, "reached_exact_qubo_optimum": gap == 0}
        for seed, gap in zip((1, 2, 3, 4), (0.1, 0.4, 0.2, 0.5))
    ]
    summary = summarize_trials(left, bootstrap_resamples=500)
    assert summary["n_trials"] == 4
    assert summary["optimizer_run_exact_qubo_hit"]["hits"] == 1
    assert (
        summary["optimizer_run_exact_qubo_hit"]["unit_of_replication"]
        == "independent optimizer run"
    )
    paired = summarize_paired_gap_difference(left, right, bootstrap_resamples=500)
    assert paired["median_difference"] < 0


def test_projected_validation_reports_voltage_and_thermal_excess():
    trials = [
        {
            "projection": {
                "projected_fixed_topology_validation": {
                    "vmin_pu": 0.88,
                    "vmax_pu": 1.05,
                    "diagnostics": {"max_current_utilization": 1.03},
                    "nonlinear_ac": None,
                }
            }
        },
        {
            "projection": {
                "projected_fixed_topology_validation": {
                    "vmin_pu": 0.95,
                    "vmax_pu": 1.12,
                    "diagnostics": {"max_apparent_power_utilization": 0.8},
                    "nonlinear_ac": {
                        "vmin_pu": 0.94,
                        "vmax_pu": 1.11,
                        "max_current_utilization": 1.07,
                    },
                }
            }
        },
    ]
    voltage, thermal = _projected_max_violations(trials)
    assert np.isclose(voltage, 0.02)
    assert np.isclose(thermal, 0.07)
