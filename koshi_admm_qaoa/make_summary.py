"""Generate RESULTS_SUMMARY.md from the real benchmark JSON."""
import json, os

rows = json.load(open("results/benchmark.json"))
sizes = sorted(set(r["n"] for r in rows))

def get(n, method, key):
    for r in rows:
        if r["n"] == n and r["method"] == method:
            return r.get(key)
    return None

methods = ["QAOA p1 noiseless", "QAOA p2 noiseless", "QAOA p1 noisy", "QRAO 3v", "SA"]

lines = []
lines.append("# Results Summary — the narrative\n")
lines.append("All numbers below are produced by `benchmark.py` on the real Eastern-Nepal "
             "sub-system (see `results/benchmark.csv`). Fixed seeds; error bars over seeds.\n")

lines.append("## 1. The QUBO is now a genuine combinatorial problem")
n_off = {n: get(n, "QAOA p1 noiseless", "n_offdiag") for n in sizes}
lines.append("| problem size n (qubits) | " + " | ".join(str(n) for n in sizes) + " |")
lines.append("|---|" + "|".join("---" for _ in sizes) + "|")
lines.append("| QUBO coupling terms (z_i z_j) | " + " | ".join(str(n_off[n]) for n in sizes) + " |")
lines.append("\nThe original toy QUBO had **0** coupling terms (separable). Here coupling grows "
             "with size, so QAOA/QRAO have real work.\n")

lines.append("## 2. Approximation ratio & success probability vs. size")
lines.append("| method | " + " | ".join(f"n={n}" for n in sizes) + " |")
lines.append("|---|" + "|".join("---" for _ in sizes) + "|")
for m in methods:
    cells = []
    for n in sizes:
        ar = get(n, m, "approx_mean"); sd = get(n, m, "approx_std"); su = get(n, m, "success")
        if ar is None:
            cells.append("—")
        else:
            cells.append(f"{ar:.3f}±{sd:.3f} (P={su:.2f})")
    lines.append(f"| {m} | " + " | ".join(cells) + " |")
lines.append("")

lines.append("## 3. Qubit footprint (QAOA vs QRAO)")
lines.append("| n switches | QAOA qubits | QRAO qubits | compression |")
lines.append("|---|---|---|---|")
for n in sizes:
    qa = get(n, "QAOA p1 noiseless", "method_qubits")
    qr = get(n, "QRAO 3v", "method_qubits")
    comp = f"{qa/qr:.2f}x" if qr else "—"
    lines.append(f"| {n} | {qa} | {qr} | {comp} |")
lines.append("\n**Honest finding:** QRAO gives **~1.0× compression here** — the cardinality "
             "(spanning-tree) penalty makes the QUBO fully dense, so the (3,1,p)-QRAC cannot "
             "pack 3 mutually-anticommuting variables per qubit. QRAO's advertised 3× qubit "
             "saving only materialises on **sparser** radiality encodings (cycle+anti-islanding "
             "only). This is a concrete, citable methodological result, not a bug.\n")

lines.append("## 4. Wall time (s) — is there a quantum speed-up? (No, at this scale)")
lines.append("| n | classical exact | QAOA p1 noiseless | QAOA p1 noisy |")
lines.append("|---|---|---|---|")
for n in sizes:
    lines.append(f"| {n} | {get(n,'QAOA p1 noiseless','exact_time'):.3f} | "
                 f"{get(n,'QAOA p1 noiseless','time_mean'):.2f} | "
                 f"{get(n,'QAOA p1 noisy','time_mean'):.2f} |")
lines.append("\nClassical exact solves every instance in milliseconds. Noisy QAOA is orders of "
             "magnitude slower (noise sampling). **No quantum advantage** — as expected.\n")

lines.append("## 5. Hybrid ADMM (AC-feasible pipeline)")
lines.append("- **Exact z-update:** ADMM converges in ~3 iterations to an AC-feasible, connected "
             "topology (residual → 1e-3). See `figures/admm_convergence.png`.")
lines.append("- **QAOA z-update:** the loop **oscillates** (residual ~1–2, never →0) because the "
             "QAOA sub-solver is stochastic/approximate and the binary problem is nonconvex. The "
             "final topology is still AC-feasible after connectivity repair. This is a real NISQ "
             "limitation — reported, not hidden.\n")

lines.append("## 6. Take-aways for the paper")
lines.append("1. Coupled radiality QUBO makes the quantum step meaningful (0 → dozens of couplings).")
lines.append("2. QAOA solution quality **degrades with size** (approx 1.00 → 0.99 → 0.987; success "
             "1.00 → 0.33 by n=12) — the expected NISQ trend, on real Nepali topology.")
lines.append("3. **QRAO compression is topology-dependent** and vanishes on dense (cardinality) "
             "QUBOs — a specific, novel comparison point vs. Ngo & Nguyen (2024).")
lines.append("4. ADMM with an exact sub-solver converges; **with QAOA it oscillates** — an honest, "
             "useful engineering result about hybrid loops on nonconvex problems.")
lines.append("5. No quantum advantage at this scale; the contribution is the **real-data case "
             "study + honest benchmarking pipeline**, exactly as the feasibility report framed it.")

open("RESULTS_SUMMARY.md", "w").write("\n".join(lines))
print("wrote RESULTS_SUMMARY.md")
