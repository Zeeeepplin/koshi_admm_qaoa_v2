# ADMM–QAOA Post-Contingency Transmission Switching — Eastern Nepal (Koshi/Kabeli)

A corrected, scaled, and honestly-benchmarked hybrid quantum–classical pipeline
for post-contingency reconfiguration/switching on a **real, meshed** Eastern-Nepal
220/132/400 kV sub-system, derived from the NEA *Transmission Directorate Year Book
2081/82* and the *NEA Power Transmission Network Map*.

This rewrites the original 5-bus prototype to fix every substantive issue raised in
the review (`koshi_admm_qaoa_review.md`).

## What changed vs. the original code
| Issue in original | Fix here |
|---|---|
| Fabricated 220 kV Dhungesanghu/Amarpur→Inaruwa line (branch 4) | **Removed.** Real redundancy now comes from the Koshi **double-circuit** lines, the real **132 kV Amarpur–Dhungesanghu** Kabeli↔Koshi tie, and the **Kushaha–Inaruwa–Duhabi** 132 kV loop. |
| Slack at Tumlingtar, Inaruwa as a load (backwards) | **Inaruwa 400/220 hub is the slack** (real strong grid tie). |
| **Separable** QUBO (0 coupling) → QAOA pointless | **Radiality-coupled QUBO** (cardinality + fundamental-cycle + anti-islanding terms) → 6→91 `z_i z_j` coupling terms; QAOA now does real combinatorial work. |
| No radiality/connectivity anywhere | Encoded (soft) in the QUBO **and** hard-validated (`is_radial`/`is_connected`) + a connectivity-repair step. |
| 3 qubits / 8 states (toy) | Scales to **14 switches / 16 buses**, with a size sweep n = 4…12 for the benchmark. |
| 17-qubit Phase-0 circuit (now deprecated) | Not used; the QUBO is built directly from network topology. |
| QAOA on a trivial z-update | QAOA/QRAO solve the coupled reconfiguration QUBO; ADMM x-update is a real AC SOCP. |
| Hand-rolled loop only | Added a **Qiskit `ADMMOptimizer`** library baseline (`admm_qiskit_baseline.py`). |
| Error mitigation enabled but never measured | Phase 3 runs **unmitigated vs mitigated** and reports the degradation. |
| Leaked IBM token | Replaced with a placeholder (kept hard-coded per request) + rotate-it warning. |

## Files
- `network_data.py`   — real meshed Eastern-Nepal dataset (provenance-documented; impedances derived from line length × conductor per-km, per-unit on 100 MVA).
- `power_model.py`    — AC branch-flow **SOCP** (fixed-z and ADMM x-update), radiality/connectivity utilities, **connectivity repair**, AC-feasibility validation.
- `qubo_builder.py`   — **radiality-coupled QUBO** (the core fix); cardinality + cycle + anti-islanding coupling.
- `solvers.py`        — exact, **QAOA** (noiseless/noisy, p-sweep), **QRAO**, **SA**, and a **true-SOCP-loss** ground truth.
- `admm_hybrid.py`    — corrected hand-rolled ADMM (AC SOCP x-update ↔ coupled-QUBO z-update) with residual logging + convergence plot.
- `admm_qiskit_baseline.py` — Qiskit `ADMMOptimizer` (3-ADMM-H) baseline.
- `benchmark.py`      — scaling + noise sweep, error bars, CSV/JSON + plots.
- `phase3_hardware.py`— IBM hardware run with **mitigated-vs-unmitigated** comparison (+ `--selftest` on a fake backend).
- `run_all.py`        — runs the whole study.

## Install
```
pip install cvxpy networkx matplotlib qiskit qiskit-algorithms qiskit-optimization qiskit-aer qiskit-ibm-runtime
```

## Run
```
python network_data.py            # print the network
python power_model.py             # AC solve of the meshed base case
python qubo_builder.py            # show the QUBO is now coupled (91 terms)
python admm_hybrid.py             # hybrid ADMM (exact + QAOA z-update)
python admm_qiskit_baseline.py    # Qiskit ADMMOptimizer baseline
python benchmark.py               # scaling + noise benchmark -> results/ + figures/
python make_admm_figure.py        # ADMM convergence comparison figure
python phase3_hardware.py --selftest   # validate Phase-3 pipeline w/o hardware
python phase3_hardware.py         # real IBM run (needs a token + queue)
```

## Honest caveats (read before citing)
- Nodal MW/MVAr loads are **engineering estimates** (the Year Book is a project
  report, not a load-flow dataset); replace with SCADA/load-flow before publishing.
- The branch-flow SOC relaxation is exact only for radial operation; meshed
  intermediate iterates are relaxations and the **final** topology is AC-revalidated.
- Radiality in the QUBO is a **soft** penalty; a connectivity-repair step guarantees
  a feasible connected final topology.
- **No quantum advantage** is claimed or observed — classical exact/SA solve these
  sizes instantly. The contribution is a working, honestly-measured hybrid pipeline
  on real Nepali topology. See `RESULTS_SUMMARY.md`.
