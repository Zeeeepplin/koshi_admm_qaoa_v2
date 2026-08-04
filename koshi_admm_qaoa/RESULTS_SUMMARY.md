# Frozen Results Summary

This file is generated from `results/benchmark.json`. The JSON artifact is the source of truth.

## Audit status

- The legacy field `success` is now named **optimizer-run exact-QUBO hit fraction**; it is not a shot-level bitstring probability.
- Infinite/disconnected SOC outcomes are stored as JSON `null` plus explicit connectivity/feasibility flags.
- SA physics scores, per-seed bitstrings, ADMM histories, QRAO-ablation records, and raw hardware evidence were not present in the release and are not reconstructed.
- The retained SOC numbers were produced by legacy continuous-model-v1 and are stale after the continuous-model-v2 correction; they remain only for discrepancy tracing.
- The historical pairwise cycle and anti-islanding terms are heuristic. An exact single-commodity-flow/tree audit is stored for every size.
- The n=4,6,8 scaling instances cannot be radial because their non-decision switches were forced closed into a cyclic mandatory subgraph.
- Hardware results remain excluded. The v2 evidence contract separates the modal sample, best-energy sample, exact-QUBO minimizer set, sampled-objective mean, and shot-level exact-set probability, and requires same-n physical validation.
- No branch contingency was passed in the retained benchmark. It is a legacy base-case artifact, not post-contingency evidence.
- `results/study_protocol.json` predeclares a separate hypothetical N-1 regeneration study; no prospective result has been back-filled.

## Benchmark rows

| n | method | objective ratio | optimizer-run exact-QUBO hit fraction | representative SOC status |
|---:|---|---:|---:|---|
| 4 | QAOA p1 noiseless | 1.000 | 1.00 | 5.834 MW, legacy-v1 connected |
| 4 | QAOA p2 noiseless | 1.000 | 1.00 | 5.834 MW, legacy-v1 connected |
| 4 | QAOA p1 noisy | 1.000 | 1.00 | 5.834 MW, legacy-v1 connected |
| 4 | QRAO 3v | 1.000 | 1.00 | 5.834 MW, legacy-v1 connected |
| 4 | SA | 1.000 | 1.00 | not recorded |
| 6 | QAOA p1 noiseless | 1.000 | 1.00 | legacy-v1 disconnected/infeasible |
| 6 | QAOA p2 noiseless | 0.999 | 0.67 | legacy-v1 disconnected/infeasible |
| 6 | QAOA p1 noisy | 1.000 | 1.00 | legacy-v1 disconnected/infeasible |
| 6 | QRAO 3v | 1.000 | 1.00 | legacy-v1 disconnected/infeasible |
| 6 | SA | 1.000 | 1.00 | not recorded |
| 8 | QAOA p1 noiseless | 1.000 | 1.00 | 5.922 MW, legacy-v1 connected |
| 8 | QAOA p2 noiseless | 1.000 | 1.00 | 5.922 MW, legacy-v1 connected |
| 8 | QAOA p1 noisy | 1.000 | 1.00 | 5.922 MW, legacy-v1 connected |
| 8 | QRAO 3v | 1.000 | 1.00 | 5.922 MW, legacy-v1 connected |
| 8 | SA | 1.000 | 1.00 | not recorded |
| 10 | QAOA p1 noiseless | 0.995 | 0.67 | legacy-v1 disconnected/infeasible |
| 10 | QAOA p2 noiseless | 0.995 | 0.67 | legacy-v1 disconnected/infeasible |
| 10 | QAOA p1 noisy | 1.000 | 1.00 | legacy-v1 disconnected/infeasible |
| 10 | QRAO 3v | 1.000 | 1.00 | legacy-v1 disconnected/infeasible |
| 10 | SA | 0.995 | 0.67 | not recorded |
| 12 | QAOA p1 noiseless | 0.987 | 0.33 | legacy-v1 disconnected/infeasible |
| 12 | QAOA p2 noiseless | 0.989 | 0.33 | legacy-v1 disconnected/infeasible |
| 12 | QAOA p1 noisy | 0.987 | 0.33 | legacy-v1 disconnected/infeasible |
| 12 | QRAO 3v | 1.000 | 1.00 | legacy-v1 disconnected/infeasible |
| 12 | SA | 0.984 | 0.00 | not recorded |
