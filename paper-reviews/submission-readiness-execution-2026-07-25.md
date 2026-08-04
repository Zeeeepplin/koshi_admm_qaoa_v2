# Submission-readiness execution record — 2026-07-25

## Outcome

The recommended simulator-only path in
`submission-readiness-plan-2026-07-24.tex` was executed. Hardware results
remain excluded. The code, prospective primary study, sensitivity study,
ADMM stopping audit, generated evidence, manuscript integration, and local
verification gates are complete.

The local scientific package is ready for author review, but formal
submission remains blocked by author-supplied declarations and an immutable
release identifier.

## Frozen prospective evidence

- Protocol revision: `post-contingency-v2`
- Protocol SHA-256:
  `006f19c8d5d8ab6a9446f7dabf9c9e2dbdc47d2d2fe3d9f1a6ffa5516e12f5b3`
- Scenario: deterministic hypothetical outage of branch 3,
  Basantapur–Inaruwa circuit A
- Primary artifact: `results/post_contingency_v1.json`
  - four methods, 30 independent optimizer runs per method
  - QAOA p1 and p2 median QUBO gap: 0.500
  - highest optimizer-run exact-QUBO hit fraction: QAOA p2, 14/30
  - all raw candidates: 0/30 connected and 0/30 radial per method
  - all declared projections succeeded and all projected candidates passed
    the series-only nonlinear AC validation
  - no projected candidate passed the declared SOC recovery certificate
  - SA median solver time: 0.008 s; QRAO median solver time: 396.378 s
  - no quantum advantage or method superiority is claimed
- Sensitivity artifact: `results/post_contingency_sensitivity_v1.json`
  - 11 predeclared variants, 30 unique seeds per variant
  - QAOA-p1 median QUBO gap range: 0.000–2.167
- ADMM artifact: `results/admm_post_contingency_v1.json`
  - three exact-z-update penalty runs and 30 QAOA-z-update seed runs
  - every run reached the 30-iteration cap
  - 0/30 QAOA–ADMM terminal configurations passed nonlinear AC validation
  - no ADMM convergence or engineering-feasibility claim is made

All prospective sources and derived outputs are covered by
`results/post_contingency_manifest.json`.

## Corrections completed

- Corrected source-informed network provenance and distinguished reported,
  planned, under-construction, hypothetical, and assumed quantities.
- Corrected and tested the QUBO-to-Ising objective convention, bit ordering,
  finite penalties, QRAO mapping/rounding, and raw-versus-projected lineage.
- Added deterministic, memory-bounded QRAO rounding and resumable seed-level
  checkpoints.
- Corrected the ADMM artifact runner to use the current
  `Network.switch_indices()` API and added regression coverage.
- Added strict prospective artifact validation, generated headline macros,
  topology/physical-validation tables, ADMM stopping table, figure, and
  manifest.
- Integrated the prospective evidence into the abstract, Results,
  limitations, conclusion, and availability statement.
- Kept the hardware campaign excluded, as recommended by the readiness plan.
- Added exact dependency locking, environment capture, license, and
  provenance/literature audit records.

## Verification record

- `study_protocol.py --check`: passed
- `post_contingency_pipeline.py --check`: passed
- `artifact_pipeline.py --check`: passed
- `pytest -q`: 34 passed
- Current JSON strict parse: 8 files passed
  - `results/benchmark_legacy.json` is excluded by design because the
    preserved historical aggregate contains non-standard `Infinity` tokens;
    the migrated authoritative `results/benchmark.json` is strict JSON.
- Clean-source provenance:
  - migration dirty flag: false
  - source-tree dirty flag: false
  - source lineage commit: `26895dcd3855f96a4efc286c140b657381e40c9f`
- Final local repository commit before this execution record:
  `a9a9064`
- LaTeX compile: passed
  - 17 A4 pages
  - no overfull boxes, undefined references, or multiply defined labels
  - only non-fatal underfull-box warnings and an upstream `algorithm.sty`
    UTF-8 replacement warning remain
- PDF visual QA: all 17 rendered pages inspected; wide tables, plots,
  equations, appendix material, URLs, and references are legible and
  unclipped.

## Remaining formal submission blockers

These items require author or repository-owner facts/credentials and were
not fabricated:

1. Author-contribution statement.
2. Funding statement.
3. Conflict-of-interest declaration.
4. Acknowledgments.
5. Target-journal generative-AI disclosure, if required.
6. Immutable archival release containing the exact source, strict artifacts,
   generated files, manifests, environment, tests, and README instructions.
7. DOI or equivalent permanent identifier, followed by updating the
   manuscript to cite that exact release.

The formal submission gate is closed until those declarations and the
immutable release are supplied.
