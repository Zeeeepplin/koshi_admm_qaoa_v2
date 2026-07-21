"""Produce the ADMM convergence comparison figure: EXACT z-update (converges)
vs QAOA z-update (oscillates -- honest NISQ limitation)."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from network_data import build_full_network
from admm_hybrid import run_admm

net = build_full_network()
rex = run_admm(net, rho=4.0, z_solver="exact", max_iter=25, verbose=False)
rqa = run_admm(net, rho=4.0, z_solver="qaoa", qaoa_reps=1, max_iter=12, verbose=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
he, hq = rex["history"], rqa["history"]
ax[0].semilogy(range(1, len(he["primal"]) + 1), he["primal"], "o-",
               label="exact z-update (converges)")
ax[0].semilogy(range(1, len(hq["primal"]) + 1), hq["primal"], "s--",
               label="QAOA z-update (oscillates)")
ax[0].set_xlabel("ADMM iteration"); ax[0].set_ylabel("primal residual ||α−z|| (log)")
ax[0].set_title("Hybrid ADMM convergence: exact vs QAOA sub-solver")
ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(range(1, len(he["loss"]) + 1), he["loss"], "o-", label="exact: AC loss")
ax[1].plot(range(1, len(hq["loss"]) + 1), hq["loss"], "s--", label="QAOA: AC loss")
ax[1].set_xlabel("ADMM iteration"); ax[1].set_ylabel("AC losses (MW)")
ax[1].set_title("x-update AC losses per iteration")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/admm_convergence.png", dpi=140)
print("exact: iters=%d final AC loss=%.3f MW connected=%s" %
      (rex["iters"], rex["validation"]["loss_mw"], rex["validation"]["connected"]))
print("qaoa : iters=%d final AC loss=%.3f MW connected=%s (repaired=%s)" %
      (rqa["iters"], rqa["validation"]["loss_mw"], rqa["validation"]["connected"], rqa["repaired"]))
print("saved figures/admm_convergence.png")
