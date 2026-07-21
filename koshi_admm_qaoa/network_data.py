"""
network_data.py
================
Real, meshed Eastern-Nepal transmission sub-system used for the ADMM-QAOA
post-contingency transmission-switching / reconfiguration study.

PROVENANCE (every element is traceable to a public source):
  [D2082] NEA Transmission/Project Management Directorate Year Book FY 2024/25
          (2081/2082).
  [MAP]   NEA / Rastriya Prasaran Grid Co. "Nepal Power Transmission Network Map".
  [DERIV] Impedances derived from published line length x conductor per-km R/X,
          converted to per-unit on a 100 MVA common base (see derive_pu()).

WHY THIS NETWORK (vs. the earlier 5-bus toy):
  * The earlier code contained a FABRICATED 220 kV Dhungesanghu/Amarpur->Inaruwa
    line (branch 4) that does not exist. It has been removed.
  * Real switchable redundancy now comes from REAL structure:
      - the Koshi Corridor is a DOUBLE-CIRCUIT 220 kV line (KC1 + KC4 second
        circuits) -> each circuit is an independently switchable parallel path
        [D2082, KC1/KC4].
      - the 132 kV Amarpur-Dhungesanghu tie links the Kabeli & Koshi corridors
        (19.13 km, double circuit) [D2082, project 8] -> a real loop.
      - the Kushaha-Inaruwa-Duhabi 132 kV lines form a real loop [D2082, Pkg C].
      - the 400/220 Inaruwa hub is the strong grid tie (3x315 MVA to the
        Hetauda-Dhalkebar-Inaruwa 400 kV backbone) [D2082] -> used as SLACK
        (this fixes the earlier backwards slack/load orientation).
  * Generation injections are REAL: Sanima Middle Tamor 73 MW evacuated at
    Dhungesanghu 220 kV [D2082, KC3 "evacuating 73 MW from Sanima Middle Tamor
    HEP at 220 kV"]; Kabeli-corridor IPP aggregate at Amarpur.

MODELLING NOTES / HONEST CAVEATS:
  * Per-unit branch-flow (DistFlow) model on a common 100 MVA base; transformers
    are represented as series pu reactances (off-nominal taps ignored).
  * Nodal MW/MVAr loads are ENGINEERING ESTIMATES for the eastern region (the
    Directorate is a project year-book, not a load-flow dataset). They are
    plausible and internally consistent but should be replaced with SCADA/load-
    flow values before publication. Each is flagged 'est'.
  * The branch-flow SOC relaxation is exact only for radial operation; during
    ADMM iterations the network may be transiently meshed, so the final topology
    is re-validated for radiality + AC feasibility (see power_model.ac_feasibility).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

S_BASE = 100.0  # MVA, common per-unit base


def zbase(kv: float) -> float:
    return (kv ** 2) / S_BASE


def derive_pu(length_km: float, r_per_km: float, x_per_km: float, kv: float):
    """Return (r_pu, x_pu) for a line, on the 100 MVA base at voltage kv."""
    zb = zbase(kv)
    return (length_km * r_per_km / zb, length_km * x_per_km / zb)


# Typical conductor per-km values (ohm/km) at the relevant voltage
#   220 kV twin ACSR Moose : R 0.030  X 0.30    [DERIV, matches D2082 Koshi values]
#   220 kV quad ACSR Moose : R 0.015  X 0.25
#   132 kV ACSR Bear       : R 0.11   X 0.40    [standard]
XFMR = {  # series reactance in pu on 100 MVA (X% * 100 / MVA_rating)
    "400/220_315": 0.12 * S_BASE / 315,   # ~0.0381  [D2082 Inaruwa 3x315 MVA]
    "220/132_160": 0.12 * S_BASE / 160,   # ~0.0750  [D2082 Inaruwa 2x160 MVA]
    "220/132_100": 0.12 * S_BASE / 100,   # 0.1200   [D2082 Tumlingtar 2x100 MVA]
    "220/132_30":  0.10 * S_BASE / 30,    # 0.3333   [D2082 Dhungesanghu 2x15 MVA]
}


@dataclass
class Bus:
    idx: int
    name: str
    kv: float
    p_load_mw: float = 0.0     # + = load, - = generation injection
    q_load_mvar: float = 0.0
    is_slack: bool = False
    note: str = ""


@dataclass
class Branch:
    idx: int
    frm: int
    to: int
    r_pu: float
    x_pu: float
    rating_mva: float
    switchable: bool
    name: str = ""
    kind: str = "line"       # line | xfmr | tie
    note: str = ""


@dataclass
class Network:
    buses: List[Bus]
    branches: List[Branch]
    name: str = "EasternNepal"

    @property
    def n_bus(self):
        return len(self.buses)

    @property
    def n_branch(self):
        return len(self.branches)

    @property
    def slack(self):
        return next(b.idx for b in self.buses if b.is_slack)

    def switch_indices(self):
        return [b.idx for b in self.branches if b.switchable]

    def edge_list(self):
        return [(b.frm, b.to) for b in self.branches]


def build_full_network() -> Network:
    """Full ~16-bus meshed Eastern-Nepal sub-system (documented above)."""
    buses = [
        Bus(0,  "Inaruwa_400",     400, 0.0,   0.0,  is_slack=True,
            note="400/220 hub & cross-border tie -> SLACK [D2082]"),
        Bus(1,  "Inaruwa_220",     220, 0.0,   0.0,  note="220 side of Inaruwa hub"),
        Bus(2,  "Basantapur_220",  220, 30.0,  10.0, note="220/132/33 [D2082]; load est"),
        Bus(3,  "Baneshwar_220",   220, 15.0,  5.0,  note="220/33 [D2082]; load est"),
        Bus(4,  "Tumlingtar_220",  220, -60.0, -12.0,note="Arun-basin IPP aggregate inj est [MAP]"),
        Bus(5,  "Dhungesanghu_220",220, -73.0, -12.0,note="Sanima Middle Tamor 73 MW [D2082 KC3]"),
        Bus(6,  "Dhalkebar_400",   400, 120.0, 40.0, note="400 kV backbone export boundary est"),
        Bus(7,  "Duhabi_132",      132, 90.0,  30.0, note="Sunsari industrial load est [MAP]"),
        Bus(8,  "Kusaha_132",      132, 25.0,  8.0,  note="132/11 [D2082]; load est"),
        Bus(9,  "Anarmani_132",    132, 40.0,  13.0, note="Jhapa load est [D2082]"),
        Bus(10, "Damak_132",       132, 35.0,  11.0, note="Jhapa load est [D2082]"),
        Bus(11, "Amarpur_132",     132, -40.0, -8.0, note="Kabeli-corridor IPP inj est [D2082]"),
        Bus(12, "Phidim_132",      132, 12.0,  4.0,  note="132/33 [D2082]; load est"),
        Bus(13, "Godak_132",       132, 18.0,  6.0,  note="132/33 [D2082]; load est"),
        Bus(14, "Itahari_132",     132, 55.0,  18.0, note="load hub est [MAP]"),
        Bus(15, "Dharan_132",      132, 45.0,  15.0, note="load est [MAP]"),
    ]

    # ---- line pu impedances (derived) ----
    r_ib, x_ib = derive_pu(57, 0.015, 0.25, 220)   # Basantapur-Inaruwa quad ~57 km [D2082 KC1]
    r_bb, x_bb = derive_pu(30, 0.030, 0.30, 220)   # Baneshwar-Basantapur twin ~30 km
    r_tb, x_tb = derive_pu(20, 0.030, 0.30, 220)   # Tumlingtar-Baneshwar twin ~20 km
    r_db, x_db = derive_pu(35, 0.030, 0.30, 220)   # Dhungesanghu-Basantapur twin 35 km [D2082 KC3]
    r_ad, x_ad = derive_pu(19.13, 0.11, 0.40, 132) # Amarpur-Dhungesanghu 132 double 19.13 km [D2082]
    r_ap, x_ap = derive_pu(58, 0.11, 0.40, 132)    # Amarpur-Phidim 132 (Kabeli II/III)
    r_pg, x_pg = derive_pu(55, 0.11, 0.40, 132)    # Phidim-Godak 132
    r_gd, x_gd = derive_pu(60, 0.11, 0.40, 132)    # Godak-Damak 132 tie
    r_dk, x_dk = derive_pu(28, 0.11, 0.40, 132)    # Duhabi-Kusaha 132 (Pkg C 28 km [D2082])
    r_ik, x_ik = derive_pu(22, 0.11, 0.40, 132)    # Inaruwa-Kusaha 132
    r_di, x_di = derive_pu(15, 0.11, 0.40, 132)    # Duhabi-Itahari 132
    r_it, x_it = derive_pu(18, 0.11, 0.40, 132)    # Itahari-Damak 132
    r_da, x_da = derive_pu(45, 0.11, 0.40, 132)    # Damak-Anarmani 132
    r_idh, x_idh = derive_pu(16, 0.11, 0.40, 132)  # Itahari-Dharan 132
    r_dl, x_dl = derive_pu(84, 0.012, 0.28, 400)   # Inaruwa-Dhalkebar 400 ~84 km [D2082]

    B = []
    def add(frm, to, r, x, mva, sw, name, kind="line", note=""):
        B.append(Branch(len(B), frm, to, r, x, mva, sw, name, kind, note))

    # 400/220 & 220/132 transformers (fixed, non-switchable)
    add(0, 1, 0.0, XFMR["400/220_315"], 315, False, "Inaruwa 400/220", "xfmr")
    add(1, 7, 0.0, XFMR["220/132_160"], 160, False, "Inaruwa 220/132", "xfmr")
    add(5, 11, 0.0 + r_ad, XFMR["220/132_30"] + x_ad, 30, True,
        "Amarpur-Dhungesanghu 132 (tie+xfmr)", "tie",
        "Kabeli<->Koshi link [D2082 proj 8]")

    # Koshi Corridor 220 kV -- DOUBLE CIRCUIT (each circuit switchable) [D2082 KC1/KC4]
    add(1, 2, r_ib, x_ib, 600, True, "Basantapur-Inaruwa ckt-A", "line")
    add(1, 2, r_ib, x_ib, 600, True, "Basantapur-Inaruwa ckt-B", "line")
    add(2, 3, r_bb, x_bb, 600, True, "Baneshwar-Basantapur ckt-A", "line")
    add(2, 3, r_bb, x_bb, 600, True, "Baneshwar-Basantapur ckt-B", "line")
    add(3, 4, r_tb, x_tb, 600, True, "Tumlingtar-Baneshwar ckt-A", "line")
    add(3, 4, r_tb, x_tb, 600, True, "Tumlingtar-Baneshwar ckt-B", "line")
    add(2, 5, r_db, x_db, 600, True, "Dhungesanghu-Basantapur ckt-A", "line")
    add(2, 5, r_db, x_db, 600, True, "Dhungesanghu-Basantapur ckt-B", "line")

    # 400 kV backbone (fixed)
    add(0, 6, r_dl, x_dl, 1600, False, "Inaruwa-Dhalkebar 400", "line",
        "Hetauda-Dhalkebar-Inaruwa 400 kV backbone [D2082]")

    # Eastern 132 kV network (mix of fixed + tie switches -> real loops)
    add(1, 8, r_ik, x_ik, 160, True,  "Inaruwa-Kusaha 132", "tie")
    add(7, 8, r_dk, x_dk, 160, True,  "Duhabi-Kusaha 132", "tie")   # Inaruwa-Duhabi-Kusaha loop
    add(7, 14, r_di, x_di, 160, False, "Duhabi-Itahari 132", "line")
    add(14, 10, r_it, x_it, 160, False, "Itahari-Damak 132", "line")
    add(10, 9, r_da, x_da, 160, False, "Damak-Anarmani 132", "line")
    add(14, 15, r_idh, x_idh, 160, True, "Itahari-Dharan 132", "tie")

    # Kabeli corridor 132 kV (Godak-Phidim-Amarpur) + tie to eastern 132 backbone
    add(11, 12, r_ap, x_ap, 160, True, "Amarpur-Phidim 132", "tie")
    add(12, 13, r_pg, x_pg, 160, False, "Phidim-Godak 132", "line")
    add(13, 10, r_gd, x_gd, 160, True, "Godak-Damak 132", "tie",
        "closes the Kabeli<->east<->Inaruwa<->Koshi<->Dhungesanghu<->Amarpur mega-loop")

    return Network(buses, B, "EasternNepal_full")


def scaled_network(n_switches: Optional[int] = None, seed: int = 0) -> Network:
    """
    Return the full network but with only the first `n_switches` switchable
    branches kept as BINARY DECISION variables; the remaining switchable
    branches are forced CLOSED (treated as fixed).  Used for the scaling sweep
    so the physical network is identical while the combinatorial size grows.
    """
    net = build_full_network()
    if n_switches is None:
        return net
    sw = [b for b in net.branches if b.switchable]
    keep = set(b.idx for b in sw[:n_switches])
    for b in net.branches:
        if b.switchable and b.idx not in keep:
            b.switchable = False  # force closed / fixed
    net.name = f"EasternNepal_s{n_switches}"
    return net


if __name__ == "__main__":
    net = build_full_network()
    print(f"{net.name}: {net.n_bus} buses, {net.n_branch} branches, "
          f"{len(net.switch_indices())} switchable")
    tot_load = sum(b.p_load_mw for b in net.buses if b.p_load_mw > 0)
    tot_gen = -sum(b.p_load_mw for b in net.buses if b.p_load_mw < 0)
    print(f"Total load  = {tot_load:.0f} MW ; total local gen = {tot_gen:.0f} MW ; "
          f"slack import = {tot_load - tot_gen:.0f} MW")
    for b in net.branches:
        print(f"  br{b.idx:2d} {b.name:32s} {b.frm:>2}->{b.to:<2} "
              f"r={b.r_pu:.4f} x={b.x_pu:.4f} sw={int(b.switchable)} {b.kind}")
