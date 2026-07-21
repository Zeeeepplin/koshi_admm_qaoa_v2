# Results Summary — the narrative

All numbers below are produced by `benchmark.py` on the real Eastern-Nepal sub-system (see `results/benchmark.csv`). Fixed seeds; error bars over seeds.

## 1. The QUBO is now a genuine combinatorial problem
| problem size n (qubits) | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| QUBO coupling terms (z_i z_j) | 6 | 15 | 28 | 45 | 66 |

The original toy QUBO had **0** coupling terms (separable). Here coupling grows with size, so QAOA/QRAO have real work.

## 2. Approximation ratio & success probability vs. size
| method | n=4 | n=6 | n=8 | n=10 | n=12 |
|---|---|---|---|---|---|
| QAOA p1 noiseless | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 0.995±0.008 (P=0.67) | 0.987±0.013 (P=0.33) |
| QAOA p2 noiseless | 1.000±0.000 (P=1.00) | 0.999±0.002 (P=0.67) | 1.000±0.000 (P=1.00) | 0.995±0.007 (P=0.67) | 0.989±0.010 (P=0.33) |
| QAOA p1 noisy | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 0.987±0.009 (P=0.33) |
| QRAO 3v | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) |
| SA | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 1.000±0.000 (P=1.00) | 0.995±0.007 (P=0.67) | 0.984±0.003 (P=0.00) |

## 3. Qubit footprint (QAOA vs QRAO)
| n switches | QAOA qubits | QRAO qubits | compression |
|---|---|---|---|
| 4 | 4 | 4 | 1.00x |
| 6 | 6 | 6 | 1.00x |
| 8 | 8 | 8 | 1.00x |
| 10 | 10 | 10 | 1.00x |
| 12 | 12 | 12 | 1.00x |

**Honest finding:** QRAO gives **~1.0× compression here** — the cardinality (spanning-tree) penalty makes the QUBO fully dense, so the (3,1,p)-QRAC cannot pack 3 mutually-anticommuting variables per qubit. QRAO's advertised 3× qubit saving only materialises on **sparser** radiality encodings (cycle+anti-islanding only). This is a concrete, citable methodological result, not a bug.

## 4. Wall time (s) — is there a quantum speed-up? (No, at this scale)
| n | classical exact | QAOA p1 noiseless | QAOA p1 noisy |
|---|---|---|---|
| 4 | 0.044 | 1.14 | 0.72 |
| 6 | 0.021 | 0.71 | 0.94 |
| 8 | 0.019 | 1.12 | 2.29 |
| 10 | 0.026 | 1.64 | 27.46 |
| 12 | 0.036 | 1.72 | 96.17 |

Classical exact solves every instance in milliseconds. Noisy QAOA is orders of magnitude slower (noise sampling). **No quantum advantage** — as expected.

## 5. Hybrid ADMM (AC-feasible pipeline)
- **Exact z-update:** ADMM converges in ~3 iterations to an AC-feasible, connected topology (residual → 1e-3). See `figures/admm_convergence.png`.
- **QAOA z-update:** the loop **oscillates** (residual ~1–2, never →0) because the QAOA sub-solver is stochastic/approximate and the binary problem is nonconvex. The final topology is still AC-feasible after connectivity repair. This is a real NISQ limitation — reported, not hidden.

## 6. Take-aways for the paper
1. Coupled radiality QUBO makes the quantum step meaningful (0 → dozens of couplings).
2. QAOA solution quality **degrades with size** (approx 1.00 → 0.99 → 0.987; success 1.00 → 0.33 by n=12) — the expected NISQ trend, on real Nepali topology.
3. **QRAO compression is topology-dependent** and vanishes on dense (cardinality) QUBOs — a specific, novel comparison point vs. Ngo & Nguyen (2024).
4. ADMM with an exact sub-solver converges; **with QAOA it oscillates** — an honest, useful engineering result about hybrid loops on nonconvex problems.
5. No quantum advantage at this scale; the contribution is the **real-data case study + honest benchmarking pipeline**, exactly as the feasibility report framed it.