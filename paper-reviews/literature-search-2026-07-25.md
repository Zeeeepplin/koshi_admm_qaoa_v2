# Literature search record — 2026-07-25

## Scope and method

The search was run on 2026-07-25 for work on quantum optimization of
power-system topology, radiality QUBOs, ADMM combined with QAOA or QRAO,
and nonlinear/physics validation. Search strings combined:

- `QAOA transmission switching power system reconfiguration`
- `QAOA transmission grid reconfiguration radiality`
- `ADMM QAOA power system reconfiguration QRAO`
- `quantum power system reconfiguration QAOA ADMM`

Publisher and primary records were preferred: IEEE/Crossref metadata,
Elsevier article records, and versioned arXiv records. Works were included
when their title or abstract addressed at least one of topology
reconfiguration/islanding, a radiality or spanning-tree formulation,
ADMM decomposition with a quantum subproblem, or explicit power-system
physics validation. The search is a targeted current review, not a claim
of an exhaustive systematic review across every bibliographic database.

## Material findings and manuscript disposition

- Silva et al., *IEEE Transactions on Power Systems* 38(5), 4559–4571
  (2023), DOI `10.1109/TPWRS.2022.3214477`, supplies a published
  minimum-loss network-reconfiguration QUBO baseline. The bibliography was
  upgraded from its provisional arXiv record.
- Ngo et al. cover ADMM–QAOA and ADMM–QRAO on IEEE 33-bus distribution
  reconfiguration. The manuscript therefore does not claim priority for
  that algorithmic combination.
- Mokhtari et al., *Sustainable Energy, Grids and Networks* 43, 101890
  (2025), DOI `10.1016/j.segan.2025.101890`, give a classical distributed
  ADMM approach with a minimum-weight rooted-arborescence subproblem. It
  is now cited to distinguish exact radiality machinery from this
  repository's heuristic historical QUBO.
- Hartmann et al., arXiv:2511.00582 (2025), formulate topology-preserving
  QAOA primitives for distribution-grid spanning trees. It is cited as
  directly relevant recent work.
- Jiang et al., arXiv:2606.15083v2 (2026), combine graph reduction,
  physics-aware constraints, QAOA, and structured post-processing for
  power-system islanding. It is cited as relevant to the explicit
  raw-versus-repaired distinction.
- Yang et al., *Electric Power Systems Research* 250, 112148 (2026), DOI
  `10.1016/j.epsr.2025.112148`, combine QAOA and ADMM for coordinated
  generation scheduling. It is cited as adjacent rather than as a
  topology-reconfiguration study.
- Morstyn and Wang's 2024 review remains the broad field map. The paper's
  contribution language is consequently bounded to an audited,
  source-informed regional test system, explicit topology/physics
  evidence separation, and reproducible claim lineage.

## Search conclusion

The search does not support claims such as “first,” “none,” or “nearly
all.” It supports a narrower positioning: the submission is an audited
case study and evidence pipeline, not a priority or quantum-advantage
claim.
