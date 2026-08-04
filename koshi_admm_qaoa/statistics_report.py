"""Predeclared descriptive statistics for regenerated stochastic trials."""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval for a binomial proportion."""
    successes, trials = int(successes), int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def bootstrap_median_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 20260724,
) -> tuple:
    """Percentile bootstrap interval for a median using a fixed RNG seed."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap input must contain at least two finite values")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, data.size, size=(resamples, data.size))
    medians = np.median(data[sample_indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(medians, [alpha, 1.0 - alpha]).tolist())


def summarize_trials(
    trials: Iterable[Mapping],
    gap_field: str = "objective_gap",
    hit_field: str = "reached_exact_qubo_optimum",
    time_field: str = "time_s",
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 20260724,
) -> dict:
    """Summarize QUBO gaps, optimizer-run exact-QUBO hits, and wall time."""
    trials = list(trials)
    if len(trials) < 2:
        raise ValueError("at least two stochastic trials are required")
    gaps = np.asarray([float(trial[gap_field]) for trial in trials])
    times = np.asarray([float(trial[time_field]) for trial in trials])
    hits = np.asarray([bool(trial[hit_field]) for trial in trials], dtype=int)
    if not np.all(np.isfinite(gaps)) or not np.all(np.isfinite(times)):
        raise ValueError("trial gaps and times must be finite")
    gap_ci = bootstrap_median_interval(
        gaps, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    time_ci = bootstrap_median_interval(
        times, resamples=bootstrap_resamples, seed=bootstrap_seed + 1
    )
    hit_ci = wilson_interval(int(hits.sum()), len(hits))
    hit_summary = {
        "hits": int(hits.sum()),
        "successes": int(hits.sum()),  # backward-compatible alias
        "proportion": float(hits.mean()),
        "wilson_95_ci": list(hit_ci),
        "unit_of_replication": "independent optimizer run",
    }
    return {
        "n_trials": len(trials),
        "objective_gap": {
            "median": float(np.median(gaps)),
            "q1": float(np.quantile(gaps, 0.25)),
            "q3": float(np.quantile(gaps, 0.75)),
            "bootstrap_95_ci_median": list(gap_ci),
            "bootstrap_method": "within-method percentile bootstrap",
            "unit_of_replication": "independent optimizer run",
        },
        "wall_time_s": {
            "median": float(np.median(times)),
            "q1": float(np.quantile(times, 0.25)),
            "q3": float(np.quantile(times, 0.75)),
            "bootstrap_95_ci_median": list(time_ci),
            "bootstrap_method": "within-method percentile bootstrap",
            "unit_of_replication": "independent optimizer run",
        },
        "optimizer_run_exact_qubo_hit": hit_summary,
        "exact_optimum_hit": hit_summary,  # backward-compatible alias
    }


def summarize_paired_gap_difference(
    left_trials: Iterable[Mapping],
    right_trials: Iterable[Mapping],
    seed_field: str = "seed",
    gap_field: str = "objective_gap",
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 20260724,
) -> dict:
    """Summarize seed-matched gap differences ``left - right``."""
    left = {int(trial[seed_field]): trial for trial in left_trials}
    right = {int(trial[seed_field]): trial for trial in right_trials}
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("paired methods must contain the same two or more seeds")
    seeds = sorted(left)
    differences = np.asarray(
        [float(left[seed][gap_field]) - float(right[seed][gap_field]) for seed in seeds]
    )
    interval = bootstrap_median_interval(
        differences, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    return {
        "definition": "left objective gap minus right objective gap",
        "seeds": seeds,
        "median_difference": float(np.median(differences)),
        "q1": float(np.quantile(differences, 0.25)),
        "q3": float(np.quantile(differences, 0.75)),
        "bootstrap_95_ci_median": list(interval),
    }
