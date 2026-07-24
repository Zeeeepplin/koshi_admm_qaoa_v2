# ADMM–QAOA Post-Contingency Transmission Switching — Eastern Nepal (Koshi/Kabeli)

A hybrid quantum–classical pipeline
for post-contingency reconfiguration/switching on a **real, meshed** Eastern-Nepal
220/132/400 kV sub-system, derived from the NEA *Transmission Directorate Year Book
2081/82* and the *NEA Power Transmission Network Map*.

## Files

- `network_data.py`   — meshed Eastern-Nepal dataset (impedances derived from line length × conductor per-km, per-unit on 100 MVA).
- `power_model.py`    — AC branch-flow **SOCP** (fixed-z and ADMM x-update), radiality/connectivity utilities, **connectivity repair**, AC-feasibility validation.
- `qubo_builder.py`   — **radiality-coupled QUBO**; cardinality + cycle + anti-islanding coupling.
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
python qubo_builder.py            # show the QUBO is now coupled
python admm_hybrid.py             # hybrid ADMM (exact + QAOA z-update)
python admm_qiskit_baseline.py    # Qiskit ADMMOptimizer baseline
python benchmark.py               # scaling + noise benchmark -> results/ + figures/
python make_admm_figure.py        # ADMM convergence comparison figure
python phase3_hardware.py --selftest   # validate Phase-3 pipeline w/o hardware
python phase3_hardware.py         # real IBM run (needs a token + queue)
```

## Notes

- Nodal MW/MVAr loads are **engineering estimates** (the Year Book is a project
  report, not a load-flow dataset); replace with SCADA/load-flow before publishing.
- The branch-flow SOC relaxation is exact only for radial operation; meshed
  intermediate iterates are relaxations and the **final** topology is AC-revalidated.
- Radiality in the QUBO is a **soft** penalty; a connectivity-repair step guarantees
  a feasible connected final topology.
- **No quantum advantage** is observed.
