"""
run_all.py — orchestrate the whole study end to end.
Skips the real hardware run (use phase3_hardware.py directly for that).
"""
import warnings; warnings.filterwarnings("ignore")

def main():
    from network_data import build_full_network
    net = build_full_network()
    print(f"[1/6] Network: {net.n_bus} buses, {net.n_branch} branches, "
          f"{len(net.switch_indices())} switchable")

    from qubo_builder import build_reconfig_qubo
    _, meta = build_reconfig_qubo(net)
    print(f"[2/6] Coupled QUBO: {meta['n_qubits']} qubits, "
          f"{meta['n_offdiag']} coupling terms")

    import power_model as pm
    r = pm.ac_feasibility(net, {k: 1 for k in net.switch_indices()})
    print(f"[3/6] Corrected validation base case (meshed): "
          f"SOC loss={r['loss_mw']:.3f} MW, connected={r['connected']}, "
          f"nonlinear AC validated={r['nonlinear_ac_validated']}")

    from admm_hybrid import run_admm
    print("[4/6] Hybrid ADMM (exact z-update):")
    run_admm(net, rho=4.0, z_solver="exact", max_iter=25)

    from benchmark import run_benchmark, make_plots
    print("[5/6] Benchmark (scaling + noise) ...")
    run_benchmark(sizes=(4, 6, 8, 10, 12), seeds=(42, 7, 123),
                  reps_list=(1, 2), shots=1024, maxiter=50,
                  ground_truth_max_size=10)
    make_plots()

    from make_summary import main as make_summary
    make_summary()
    print("[6/6] Done. See results/ and figures/ and RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
