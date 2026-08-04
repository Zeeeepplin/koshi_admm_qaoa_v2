# Comprehensive submission-readiness execution report

Date: 2026-07-26
Workspace: `C:\Users\lenovo\Documents\testing research roshan sir`
Branch: `codex/submission-readiness`
Audit baseline commit: `92142e39880eb3ce938eace4bf22ed0616086572`

## 1. Executive summary

The simulator-only path recommended by
`paper-reviews/submission-readiness-plan-2026-07-24.tex` was executed.
The work covered the source code, mathematical formulation, network-data
provenance, experimental protocol, statistical definitions, long-running
experiments, generated artifacts, manuscript claims, reproducibility checks,
and visual inspection of the compiled paper.

The completed work materially changed the scientific position of the paper:

- the historical benchmark is now described as a base-case surrogate-QUBO
  scaling artifact, not as post-contingency restoration evidence;
- the new branch-3 study is explicitly a deterministic hypothetical scenario,
  not a recorded Nepal Electricity Authority event;
- raw solver outputs and projected topologies are archived and reported
  separately;
- no raw stochastic-solver output is connected or radial;
- all reported physical feasibility in the primary study follows an explicit
  spanning-tree projection;
- the series-only nonlinear AC validation accepts all projected primary
  candidates, while the declared SOC recovery certificate accepts none;
- the ADMM-inspired runs do not satisfy their residual stopping test and do
  not support a convergence claim;
- the classical simulated-annealing baseline is much faster than the quantum
  simulations, so no quantum advantage is observed or claimed;
- unsupported hardware, ablation, and Qiskit `ADMMOptimizer` claims remain
  excluded.

The local scientific package is complete and auditable. Formal journal
submission remains blocked only by author-controlled declarations and an
immutable DOI-bearing release.

## 2. Scope and governing decisions

### 2.1 Submission path selected

The recommended path was used:

- execute the frozen simulator protocol;
- generate the primary, sensitivity, and ADMM artifacts;
- retain the hardware evidence contract as reproducibility guidance;
- exclude hardware performance results because no validated repeated
  multi-date QPU campaign exists;
- regenerate every numerical table, figure, and headline value from validated
  machine-readable artifacts.

### 2.2 Evidence boundaries enforced

The following boundaries are now explicit throughout the code and paper:

1. reported utility assets versus planned, under-construction, hypothetical,
   or assumed network data;
2. historical base-case results versus the new hypothetical branch-3
   contingency;
3. raw QUBO outputs versus projected topologies;
4. heuristic QUBO topology penalties versus exact radiality;
5. SOC relaxation/recovery versus nonlinear AC power-flow validation;
6. optimizer-run exact-QUBO hit fraction versus shot-level sampling
   probability;
7. simulator timing versus QPU access or queue time;
8. ADMM-inspired decomposition versus a proven convergent ADMM method.

## 3. Repository, environment, and release engineering

The following repository-level work was completed:

- created and used branch `codex/submission-readiness`;
- added an MIT `LICENSE`;
- added exact dependency locking in `requirements-lock.txt`;
- recorded Python, operating system, CPU, memory, package versions, Git
  commit, working-tree state, and timing boundaries in
  `koshi_admm_qaoa/results/environment.json`;
- documented simulator execution and evidence-generation commands in the
  project README;
- separated generated files, results, figures, and temporary execution
  products;
- regenerated the authoritative legacy artifact from a clean source state;
- recorded SHA-256 hashes for legacy and post-contingency source/derived
  artifacts in independent manifests;
- preserved the original non-standard legacy aggregate only for provenance.

The clean-generation record reports:

- legacy migration dirty flag: `false`;
- source-tree dirty flag: `false`;
- source-lineage commit:
  `26895dcd3855f96a4efc286c140b657381e40c9f`.

## 4. Network model and provenance corrections

The network-data audit corrected unsupported or ambiguous statements and
classified every branch source:

- Inaruwa transformer capacity is represented as three 315 MVA units,
  totaling 945 MVA;
- the two 160 MVA Inaruwa units total 320 MVA;
- the previously asserted 2 x 15 MVA 220/132 kV attribution was removed;
- the Amarpur-Dhungesanghu 100 MVA 220/132 kV interface is explicitly
  hypothetical;
- planned KC1 circuit B and the under-construction
  Amarpur-Dhungesanghu tie are not described as currently operated assets;
- the resulting network is called source-informed, not an as-operated utility
  model;
- nodal demand, reactive demand, impedances, ratings, taps, and uncertain
  segment lengths are labeled as assumptions where source data are absent.

The supporting audit records are:

- `paper-reviews/network-provenance-audit-2026-07-25.md`;
- `koshi_admm_qaoa/generated/branch_provenance_table.tex`.

## 5. Mathematical and physical-model corrections

### 5.1 QUBO and Ising conventions

The implementation and manuscript now agree on:

- minimization objective sense;
- finite penalty values;
- binary-to-Pauli mapping
  \(z_\ell=(1-Z_\ell)/2\);
- QuadraticProgram variable order and physical branch order;
- exact QUBO offset handling;
- complete enumeration of global QUBO minimizers;
- distinction between the exact surrogate-QUBO optimum and a hard-radial
  reference solution.

### 5.2 Topology logic

The audit added or corrected:

- exact connectedness and radiality evaluation;
- a flow-based spanning-tree MILP reference;
- a cycle-safe projection with a complete action log;
- explicit raw and projected bitstrings and objective values;
- proof by exhaustive enumeration that historical pairwise cycle and
  anti-islanding terms are heuristic rather than hard topology constraints;
- proof that the historical forced-closed scaling construction makes a
  spanning tree impossible at \(n=4,6,8\);
- proof that none of the historical QUBO minimizers at \(n=10,12\) is radial.

### 5.3 Continuous model and validation

Continuous-model-v2 corrected:

- fixed-power-factor active/reactive load shedding;
- branch and transformer ratings;
- transformer tap treatment;
- slack active/reactive balance;
- topology-conditioned recovery;
- nonlinear AC validation and residual reporting;
- voltage and thermal limit diagnostics.

The paper no longer reuses legacy continuous-model-v1 losses as current
results. It also states that the nonlinear validation is a series-only
power-flow feasibility test, not an AC-OPF optimality certificate or a full
utility-model validation.

## 6. Solver and execution engineering

### 6.1 QAOA

The prospective QAOA path now records:

- depth \(p\);
- seed;
- shots;
- COBYLA limit and tolerance;
- seeded initial point;
- optimal point;
- optimizer trace;
- returned bitstring in variable order;
- raw QUBO objective and gap;
- topology and physical validation;
- projection actions and projected validation;
- source commit and package versions.

### 6.2 QRAO

QRAO execution was made deterministic and memory bounded:

- the MagicRounding basis RNG is explicitly seeded;
- primitive sampling is seeded;
- circuits are evaluated in batches of 256;
- independent workers use process isolation;
- worker count was restored to the frozen four-worker protocol after the
  batching fix;
- a deterministic regression test covers the rounding path;
- seed-level checkpoints allow safe resumption.

The retained historical QRAO result still uses one qubit per binary variable.
No unsupported qubit-compression or ablation claim was restored.

### 6.3 Sensitivity and checkpointing

Primary QRAO, one-factor sensitivity, and ADMM studies checkpoint completed
seeds. Re-running a script reuses only records whose protocol, seed set,
configuration, and source metadata satisfy the archived contract.

### 6.4 ADMM-inspired runner

The ADMM artifact runner was corrected to use the current
`Network.switch_indices()` API rather than the removed
`Network.switch_branches` attribute. A regression test covers current switch
discovery. Complete primal and dual residual histories, configurations,
termination reasons, final raw/projected topologies, and validation records
are archived.

## 7. Frozen experimental protocol

Protocol revision: `post-contingency-v2`

Protocol SHA-256:
`006f19c8d5d8ab6a9446f7dabf9c9e2dbdc47d2d2fe3d9f1a6ffa5516e12f5b3`

Primary scenario:

- forced-open branch: 3;
- physical branch: Basantapur-Inaruwa circuit A;
- scenario meaning: deterministic hypothetical N-1 case;
- decision variables: 13 remaining switches;
- stochastic repetitions: 30 independent optimizer runs per method;
- QAOA depths: 1 and 2;
- QAOA/QRAO shots: 4096;
- optimizer: COBYLA, 100 iterations, tolerance \(10^{-4}\);
- SA: 1000 iterations, \(T_0=10\), cooling factor 0.995;
- bootstrap: 10,000 within-method resamples with fixed seed 20260724;
- binary-rate intervals: Wilson 95 percent intervals;
- no paired inference from reused numeric seed labels.

## 8. Executed experiments and results

### 8.1 Primary post-contingency benchmark

Artifact: `koshi_admm_qaoa/results/post_contingency_v1.json`
Size: 41,388,998 bytes
SHA-256:
`f7be4479b446dbdd63fcb7a6b391c5f9c2c5e33deb36b6df8e1e39cf6dc0fdae`

| Method | Runs | Median QUBO gap | 95% bootstrap CI | Exact-QUBO hits | Median solver time |
|---|---:|---:|---:|---:|---:|
| QAOA p1 noiseless | 30 | 0.500 | [0.250, 0.682] | 10/30 | 7.586 s |
| QAOA p2 noiseless | 30 | 0.500 | [0.000, 0.500] | 14/30 | 12.078 s |
| QRAO 3v | 30 | 0.682 | [0.250, 0.682] | 10/30 | 396.378 s |
| SA | 30 | 0.682 | [0.000, 0.682] | 12/30 | 0.008 s |

Topology and validation result for every method:

- raw connected rate: 0/30;
- raw radial rate: 0/30;
- projection success: 30/30;
- declared SOC recovery certificate: 0/30;
- series-only nonlinear AC validation after projection: 30/30;
- median validated projected loss: 4.101 MW.

Interpretation:

- QAOA p1 and p2 tie on the median gap;
- QAOA p2 has the largest point estimate of the optimizer-run exact-QUBO hit
  fraction, but uncertainty intervals overlap;
- SA is orders of magnitude faster;
- physical feasibility is attributable to explicit projection and validation,
  not to the raw QUBO output;
- no method-superiority or quantum-advantage claim is supported.

### 8.2 One-factor sensitivity study

Artifact:
`koshi_admm_qaoa/results/post_contingency_sensitivity_v1.json`
Size: 12,688,238 bytes
SHA-256:
`0b940fdac1e5279896a4761ab718fd43a5ae71b94d0fce03783e73bc64ea1fd7`

Eleven predeclared variants contain 30 unique QAOA-p1 seeds each. Median
within-variant gaps range from 0.000 to 2.167. The strongest point estimates
of the exact-QUBO hit fraction occur for loss bias 2.5 (27/30) and
\(\lambda_{\mathrm{cyc}}=3\) (26/30).

These rows change either the QUBO or the optimizer budget. Therefore their
gaps and hit rates are relative to different row-specific optima and are
reported as tuning sensitivity, not as a cross-objective ranking. The full
11-row table is generated into
`generated/post_contingency_sensitivity_table.tex`.

### 8.3 ADMM stopping study

Artifact:
`koshi_admm_qaoa/results/admm_post_contingency_v1.json`
Size: 1,474,750 bytes
SHA-256:
`7f7888b82474ff5c73add0a4af890a9335d793626970135055da4936fff9c130`

Executed records:

- exact-z update at \(\rho=1.5\), 3.0, and 6.0;
- 30 independent QAOA-z-update runs at \(\rho=3\);
- 30-iteration cap;
- primal and dual tolerances of \(10^{-2}\).

Results:

- every exact-z run terminated at `maximum_iterations`;
- all 30 QAOA-z runs terminated at `maximum_iterations`;
- median QAOA final primal residual: 2.219;
- median QAOA final dual residual: 7.937;
- 0/30 QAOA-ADMM terminal configurations passed nonlinear AC validation.

The evidence supports a stopping-test failure under the frozen configuration,
not an ADMM convergence or restoration-feasibility claim.

## 9. Statistical and reporting corrections

The following terminology and inferential corrections were applied:

- the historical field called "success probability" is identified as an
  optimizer-run QUBO-minimum hit fraction;
- shots are not treated as independent optimizer runs;
- repeated numeric seed labels across methods do not create paired
  observations;
- method summaries use within-method medians and percentile-bootstrap
  intervals;
- binary proportions use Wilson intervals;
- objective comparisons use the offset-invariant QUBO gap;
- sensitivity results explicitly state when the objective changes;
- unavailable values are excluded or marked not reported rather than copied
  between methods;
- wall-time inclusion and exclusion boundaries are stated;
- no statistical significance or superiority claim is inferred from
  overlapping descriptive intervals.

## 10. Generated evidence and manuscript integration

The post-contingency pipeline now generates and hashes:

- `generated/post_contingency_numbers.tex`;
- `generated/post_contingency_results_table.tex`;
- `generated/post_contingency_validation_table.tex`;
- `generated/post_contingency_sensitivity_table.tex`;
- `generated/post_contingency_admm_table.tex`;
- `figures/post_contingency_objective_gaps.png`;
- `results/post_contingency_manifest.json`.

`main_2.tex` was updated in the following locations:

- abstract: scenario, run counts, median gap, hit fraction, raw topology
  failure, projection dependence, SOC/AC distinction, ADMM stopping result,
  and no-advantage statement;
- methods: executed protocol, estimands, seeds, optimizer settings,
  confidence intervals, ADMM rules, software versions, process isolation,
  memory boundary, and reproducibility gates;
- Results: generated primary table, complete seed plot, validation table,
  sensitivity table, ADMM stopping table, and conservative interpretation;
- Discussion: claim-lineage lessons, projection dependence, validation scope,
  parameter sensitivity, and negative ADMM evidence;
- Limitations: source-informed rather than operational network, hypothetical
  outage, series-only nonlinear validation, absence of hardware evidence, and
  lack of quantum advantage;
- Conclusion: supported findings only;
- Code and data availability: authoritative artifacts, pipelines, manifests,
  environment record, and evidence-capture scripts.

Wide tables are kept adjacent to the relevant Results block with a float
barrier. The post-contingency plot was resized for a legible single-column
layout.

## 11. Claim disposition matrix

| Claim area | Final disposition |
|---|---|
| Historical rows are post-contingency | Rejected; they are base-case surrogate-QUBO records |
| Historical QUBO enforces radiality | Rejected by exhaustive audit |
| QAOA "success" is a shot probability | Rejected; it is an optimizer-run hit fraction |
| Raw post-contingency solver outputs are valid topologies | Rejected; 0/30 connected/radial for every method |
| Projected prospective candidates pass nonlinear AC | Supported for the declared series-only check, 30/30 per method |
| Projected candidates pass the declared SOC certificate | Rejected, 0/30 per method |
| ADMM converges | Rejected under the frozen 30-iteration protocol |
| Quantum method is faster or superior | Rejected; SA is much faster and intervals overlap |
| QRAO reduces qubits in the retained record | Rejected; retained record is one qubit per variable |
| Hardware performance is validated | Excluded; no compliant repeated QPU evidence package |
| Network is an as-operated Nepal utility model | Rejected; it is source-informed and assumption-bearing |

## 12. Verification completed

The final verification gate includes:

- `study_protocol.py --check`;
- `post_contingency_pipeline.py --check`;
- `artifact_pipeline.py --check`;
- full `pytest -q`;
- strict JSON parsing of all current authoritative artifacts;
- clean-source provenance checks;
- LaTeX compilation with bibliography regeneration;
- log search for overfull boxes, undefined references, and multiply defined
  labels;
- Poppler rendering and visual inspection of every PDF page.

The latest verified baseline produced:

- 34 passing tests;
- 8 current strict-JSON files;
- 18 A4 PDF pages;
- final PDF size 4,424,527 bytes and SHA-256
  `005c451948370126949f40dea5d11dcaa32672f1d42aec00bc09cd946102be46`;
- no overfull boxes;
- no undefined references or citations;
- no multiply defined labels;
- only non-fatal dependency deprecation warnings, underfull-box warnings, and
  an upstream `algorithm.sty` UTF-8 replacement warning.

## 13. Important defects found during execution

The execution phase caught defects that a manuscript-only review would not
have exposed:

1. QRAO rounding did not pass the archived seed into the MagicRounding basis
   RNG.
2. QRAO attempted to hold thousands of circuits in memory at once.
3. long studies lacked safe seed-level resume behavior.
4. the ADMM artifact runner referenced the removed
   `Network.switch_branches` API.
5. the generated Matplotlib label used invalid unbraced math-text syntax.
6. generated ADMM termination strings did not escape underscores for LaTeX.
7. the primary results table exceeded the two-column width.
8. wide post-contingency floats drifted far from their Results discussion.
9. the sensitivity artifact was validated but not fully tabulated in the
   manuscript.

Each defect has been corrected and covered by execution, validation, or a
regression test.

## 14. Reproduction commands

From `koshi_admm_qaoa` in the pinned environment:

```powershell
python study_protocol.py --check
python benchmark.py --post-contingency
python run_post_contingency_sensitivity.py
python make_post_contingency_admm.py
python post_contingency_pipeline.py
python post_contingency_pipeline.py --check
python artifact_pipeline.py --check
python -m pytest -q
```

The long-running scripts are resumable. A rerun must not be interpreted as a
new independent experiment unless the protocol revision, seeds, and artifact
identity are changed deliberately.

## 15. Milestone commit history

| Commit | Milestone |
|---|---|
| `b84bba8` | Bounded and seeded QRAO MagicRounding |
| `4c71a53` | Restored frozen four-worker QRAO execution |
| `8fe2bc3` | Clarified bounded reference sampling |
| `2ba4fff` | Archived the primary post-contingency benchmark |
| `6e1b54f` | Archived the 11-variant sensitivity sweep |
| `d185ce2` | Fixed ADMM network switch discovery |
| `cf64d53` | Archived the ADMM stopping study |
| `6b80b69` | Integrated post-contingency evidence into the manuscript |
| `26895dc` | Kept generated evidence with its Results discussion |
| `a9a9064` | Refreshed final reproducibility manifests |
| `92142e3` | Added the concise execution record |
| `9f4ff30` | Added this comprehensive report and the full sensitivity audit |
| `725d538` | Refreshed strict artifacts with clean-source provenance |

## 16. Remaining formal submission blockers

The following facts or external actions require the authors or repository
owner and were not fabricated:

1. CRediT or journal-specific author-contribution statement.
2. Funding statement, including grant numbers if applicable.
3. Conflict-of-interest declaration.
4. Acknowledgments.
5. Target-journal generative-AI disclosure, if required.
6. Confirmation of the exact target journal and its declaration format.
7. Immutable release containing the exact source, raw strict artifacts,
   generated products, manifests, environment file, tests, README execution
   instructions, and commit hash.
8. DOI or equivalent permanent identifier.
9. Final update of the manuscript availability statement to cite that exact
   immutable release.

These are the only remaining submission-gate items. They are external to the
scientific and local reproducibility work completed here.

## 17. Final assessment

The codebase and manuscript now support a conservative, auditable paper about
the behavior and limitations of an ADMM-inspired quantum-classical workflow
on a source-informed Eastern Nepal test system. The strongest scientific
result is not quantum superiority. It is the evidence lineage showing that:

- surrogate-QUBO quality does not establish topology validity;
- projection can dominate the physical meaning of returned candidates;
- SOC and nonlinear AC acceptance must be reported as distinct tests;
- a failed stopping test is a reportable ADMM result;
- classical baselines remain decisive at this scale;
- reproducible negative results are more defensible than unsupported
  performance claims.

Subject to the author declarations and immutable release listed above, the
local package is ready for final author approval and journal formatting.
