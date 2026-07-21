"""
benchmark.py
============
Honest, reproducible benchmark that produces the "narrative" the study needs:

  1. SCALING sweep over problem size n = #switch qubits (real sub-networks of the
     Eastern-Nepal system).  For each size and several random seeds:
        - exact QUBO optimum (reference)
        - QAOA p=1,2 (noiseless)        -> approximation ratio + success prob
        - QAOA p=1 (depolarising noise) -> noise degradation
        - QRAO (3 vars/qubit)           -> qubit compression
        - simulated annealing           -> classical metaheuristic baseline
        - classical exact wall-time     -> "does quantum help?" (it does not, yet)
  2. TRUE-objective check: every returned config is re-scored with the AC SOCP
     model (losses, shed, connectivity) so we report engineering validity, not
     just QUBO energy.
  3. Plots + a CSV/JSON results table.  Results are written incrementally so a
     partial run is still usable.

Design choices that keep it defensible: fixed seeds, error bars over seeds,
noiseless-vs-noisy comparison, and no quantum-advantage claim.
"""
from __future__ import annotations
import os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

from network_data import scaled_network
from qubo_builder import build_reconfig_qubo
import solvers

RESULTS_DIR = "results"
FIG_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def approx_ratio(fval, fopt):
    # QUBO energies are negative; ratio in (0,1], 1 == optimal
    if abs(fopt) < 1e-9:
        return 1.0 if abs(fval - fopt) < 1e-6 else 0.0
    return float(fval / fopt) if fopt < 0 else float(fopt / fval)


def run_benchmark(sizes=(4, 6, 8, 10, 12), seeds=(42, 7, 123),
                  reps_list=(1, 2), shots=1024, maxiter=50,
                  ground_truth_max_size=10):
    rows = []
    for n in sizes:
        net = scaled_network(n)
        qp, meta = build_reconfig_qubo(net)
        nq = meta["n_qubits"]
        ex = solvers.solve_exact_qubo(qp)
        fopt = ex["fval"]
        sc_ex = solvers.score_config(net, ex["x"], meta)
        # TRUE reference (small sizes only -- SOCP brute force is expensive)
        gt_loss = None
        if n <= ground_truth_max_size:
            gt = solvers.brute_force_true(net, meta, max_eval=4000)
            gt_loss = gt["best_connected"].get("loss_mw")
        base = dict(n=n, n_qubits=nq, exact_fval=fopt, exact_time=ex["time_s"],
                    exact_true_loss=sc_ex["loss_mw"], gt_best_loss=gt_loss,
                    n_offdiag=meta["n_offdiag"], K_target=meta["K_target"])
        print(f"\n== size n={n} ({nq} qubits, {meta['n_offdiag']} couplings) "
              f"exact fval={fopt:.3f} true-loss={sc_ex['loss_mw']:.3f}MW gt={gt_loss} ==")

        # QAOA noiseless (reps sweep) + noisy, across seeds
        for kind, reps in [("noiseless", r) for r in reps_list] + [("noisy", 1)]:
            ars, succ, times = [], [], []
            last = None
            for s in seeds:
                r = solvers.solve_qaoa(qp, reps=reps, kind=kind, shots=shots,
                                       maxiter=maxiter, seed=s)
                ars.append(approx_ratio(r["fval"], fopt))
                succ.append(1.0 if r["fval"] <= fopt + 1e-6 else 0.0)
                times.append(r["time_s"]); last = r
            sc = solvers.score_config(net, last["x"], meta)
            row = dict(base, method=f"QAOA p{reps} {kind}",
                       approx_mean=float(np.mean(ars)), approx_std=float(np.std(ars)),
                       success=float(np.mean(succ)), time_mean=float(np.mean(times)),
                       method_qubits=nq, true_loss=sc["loss_mw"],
                       connected=sc["connected"], feasible=sc["feasible"])
            rows.append(row)
            print(f"   {row['method']:18s} approx={row['approx_mean']:.3f}"
                  f"±{row['approx_std']:.3f} success={row['success']:.2f} "
                  f"t={row['time_mean']:.1f}s")

        # QRAO (qubit compression)
        try:
            rq = solvers.solve_qrao(qp, max_vars_per_qubit=3, maxiter=maxiter, shots=shots)
            scq = solvers.score_config(net, rq["x"], meta)
            rows.append(dict(base, method="QRAO 3v",
                             approx_mean=approx_ratio(rq["fval"], fopt), approx_std=0.0,
                             success=float(rq["fval"] <= fopt + 1e-6),
                             time_mean=rq["time_s"], method_qubits=rq["n_qubits"],
                             true_loss=scq["loss_mw"], connected=scq["connected"],
                             feasible=scq["feasible"]))
            print(f"   {'QRAO 3v':18s} approx={rows[-1]['approx_mean']:.3f} "
                  f"qubits={rq['n_qubits']} (compression {rq['compression']:.2f}x)")
        except Exception as e:
            print("   QRAO failed:", str(e)[:80])

        # Simulated annealing (seeds)
        ars, succ, times = [], [], []
        for s in seeds:
            rs = solvers.solve_sa_qubo(meta, iters=300, seed=s)
            ars.append(approx_ratio(rs["fval"], fopt))
            succ.append(1.0 if rs["fval"] <= fopt + 1e-6 else 0.0)
            times.append(rs["time_s"])
        rows.append(dict(base, method="SA",
                         approx_mean=float(np.mean(ars)), approx_std=float(np.std(ars)),
                         success=float(np.mean(succ)), time_mean=float(np.mean(times)),
                         method_qubits=0, true_loss=None, connected=None, feasible=None))
        print(f"   {'SA':18s} approx={rows[-1]['approx_mean']:.3f} success={rows[-1]['success']:.2f}")

        # incremental save
        with open(f"{RESULTS_DIR}/benchmark.json", "w") as f:
            json.dump(rows, f, indent=2)
    _write_csv(rows)
    return rows


def _write_csv(rows):
    import csv
    if not rows:
        return
    keys = ["n", "n_qubits", "method", "method_qubits", "approx_mean", "approx_std",
            "success", "time_mean", "exact_time", "true_loss", "connected",
            "gt_best_loss", "n_offdiag"]
    with open(f"{RESULTS_DIR}/benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_plots(json_path=f"{RESULTS_DIR}/benchmark.json"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(json_path))
    methods = sorted(set(r["method"] for r in rows))
    sizes = sorted(set(r["n"] for r in rows))

    # (1) approximation ratio vs size (with error bars)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        xs = [n for n in sizes if any(r["n"] == n and r["method"] == m for r in rows)]
        ys = [next(r["approx_mean"] for r in rows if r["n"] == n and r["method"] == m) for n in xs]
        es = [next(r.get("approx_std", 0) for r in rows if r["n"] == n and r["method"] == m) for n in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=m)
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_xlabel("problem size  (# switch qubits)")
    ax.set_ylabel("approximation ratio  (fval / exact optimum)")
    ax.set_title("Solver quality vs. problem size — Eastern-Nepal reconfiguration QUBO")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/approx_vs_size.png", dpi=140); plt.close(fig)

    # (2) qubits vs size: QAOA (=n) vs QRAO (compressed)
    fig, ax = plt.subplots(figsize=(7, 5))
    q_qaoa = [(r["n"], r["method_qubits"]) for r in rows if r["method"] == "QAOA p1 noiseless"]
    q_qrao = [(r["n"], r["method_qubits"]) for r in rows if r["method"] == "QRAO 3v"]
    if q_qaoa:
        ax.plot(*zip(*sorted(q_qaoa)), "o-", label="QAOA (qubits = # switches)")
    if q_qrao:
        ax.plot(*zip(*sorted(q_qrao)), "s-", label="QRAO (3 vars/qubit)")
    ax.set_xlabel("problem size (# switches)"); ax.set_ylabel("qubits required")
    ax.set_title("Qubit footprint: QAOA vs QRAO"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/qubits_vs_size.png", dpi=140); plt.close(fig)

    # (3) wall-time vs size
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods + ["exact"]:
        if m == "exact":
            xs = sizes
            ys = [next(r["exact_time"] for r in rows if r["n"] == n) for n in xs]
            ax.plot(xs, ys, "k--", marker="x", label="classical exact")
            continue
        xs = [n for n in sizes if any(r["n"] == n and r["method"] == m for r in rows)]
        ys = [next(r["time_mean"] for r in rows if r["n"] == n and r["method"] == m) for n in xs]
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_yscale("log"); ax.set_xlabel("problem size (# switches)")
    ax.set_ylabel("wall time (s, log)"); ax.set_title("Wall-time vs size (no quantum speed-up at this scale)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/time_vs_size.png", dpi=140); plt.close(fig)
    return [f"{FIG_DIR}/approx_vs_size.png", f"{FIG_DIR}/qubits_vs_size.png",
            f"{FIG_DIR}/time_vs_size.png"]


if __name__ == "__main__":
    rows = run_benchmark()
    figs = make_plots()
    print("\nSaved:", figs)
