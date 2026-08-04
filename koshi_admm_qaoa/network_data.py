"""
network_data.py
================
Source-informed, meshed Eastern-Nepal test system used for the ADMM-QAOA
post-contingency transmission-switching / reconfiguration study. It is not an
as-operated NEA network model.

PROVENANCE:
  [D2082] NEA Transmission/Project Management Directorate Year Book FY 2024/25
          (2081/2082).
  [MAP]   NEA / Rastriya Prasaran Grid Co. "Nepal Power Transmission Network Map".
  [DERIV] Impedances derived from the documented or assumed length multiplied
          by assumed conductor per-km R/X, converted to per-unit on a 100 MVA
          common base (see derive_pu()).

The model combines reported in-service assets, planned projects, and explicitly
hypothetical interfaces to form a reproducible topology experiment. Selected
routes, aggregate lengths, and equipment multiplicities are source-informed.
Conductor electrical parameters, segment lengths, branch ratings, transformer
short-circuit reactances, operating taps, and all nodal operating-point values
are engineering assumptions. Each Branch record labels that distinction.

WHY THIS NETWORK (vs. the earlier 5-bus toy):
  * The earlier code contained a FABRICATED 220 kV Dhungesanghu/Amarpur->Inaruwa
    line (branch 4) that does not exist. It has been removed.
  * Switchable redundancy is a source-informed experimental construction:
      - KC1 was reported with one circuit in service; its second circuit was
        listed in planned package KC5. Both are modeled as independently
        switchable to study a prospective double-circuit configuration.
      - the 132 kV Amarpur-Dhungesanghu link (19.13 km, double circuit) was
        reported under construction. It is modeled as a prospective tie, with
        an explicitly assumed voltage-interface equivalent.
      - the Kushaha-Inaruwa-Duhabi 132 kV lines form a real loop [D2082, Pkg C].
      - the 400/220 Inaruwa hub is the strong grid tie (3x315 MVA to the
        Hetauda-Dhalkebar-Inaruwa 400 kV backbone) [D2082] -> used as SLACK
        (this fixes the earlier backwards slack/load orientation).
  * Generation injections are REAL: Sanima Middle Tamor 73 MW evacuated at
    Dhungesanghu 220 kV [D2082, KC3 "evacuating 73 MW from Sanima Middle Tamor
    HEP at 220 kV"]; Kabeli-corridor IPP aggregate at Amarpur.

MODELLING NOTES / HONEST CAVEATS:
  * Per-unit branch-flow (DistFlow) model on a common 100 MVA base. Nominal
    transformer ratios are absorbed by the bus voltage bases; an explicit
    off-nominal tap parameter is retained for every branch. Published tap
    positions were unavailable, so the present transformers use tap = 1.0 pu
    and are flagged as assumptions that require source replacement.
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


# Assumed typical conductor per-km values (ohm/km) at the relevant voltage
#   220 kV twin ACSR Moose : R 0.030  X 0.30
#   220 kV quad ACSR Moose : R 0.015  X 0.25
#   132 kV ACSR Bear       : R 0.11   X 0.40
XFMR = {  # parallel-bank equivalent reactance on the common 100 MVA base
    "400/220_3x315": 0.12 * S_BASE / (3 * 315),
    "220/132_2x160": 0.12 * S_BASE / (2 * 160),
    "assumed_220/132_100": 0.12 * S_BASE / 100,
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
    tap_ratio_pu: float = 1.0  # real off-nominal from-side tap; nominal ratio is in bus bases
    length_km: Optional[float] = None
    physical_units: int = 1
    parameter_status: str = "engineering estimate"
    source_ref: str = "model assumption"

    @property
    def rating_pu(self) -> float:
        return self.rating_mva / S_BASE

    @property
    def current_limit_sq_pu(self) -> float:
        # With voltage bases matched to each branch's nominal kV, I/I_base = S/S_base.
        return self.rating_pu ** 2


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
    """Full 16-bus source-informed Eastern-Nepal test system."""
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
    # KC1 is reported as 106--107 route-km in aggregate; the following segment
    # split (57 + 30 + 20 km) is a modeling assumption.
    r_ib, x_ib = derive_pu(57, 0.015, 0.25, 220)
    r_bb, x_bb = derive_pu(30, 0.030, 0.30, 220)
    r_tb, x_tb = derive_pu(20, 0.030, 0.30, 220)
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
    def add(
        frm, to, r, x, mva, sw, name, kind="line", note="", tap=1.0,
        length_km=None, physical_units=1, parameter_status="engineering estimate",
        source_ref="model assumption",
    ):
        if tap <= 0:
            raise ValueError(f"branch tap must be positive, got {tap}")
        B.append(
            Branch(
                len(B), frm, to, r, x, mva, sw, name, kind, note, tap,
                length_km, physical_units, parameter_status, source_ref,
            )
        )

    # 400/220 & 220/132 transformers (fixed, non-switchable)
    add(0, 1, 0.0, XFMR["400/220_3x315"], 945, False, "Inaruwa 400/220", "xfmr",
        "Three 315-MVA units represented as one parallel equivalent; nominal ratio "
        "is absorbed by bus bases; off-nominal tap assumed 1.0 pu",
        tap=1.0, physical_units=3, parameter_status="source-derived + assumed",
        source_ref="[D2082] multiplicity/rating; 12% reactance and tap assumed")
    add(1, 7, 0.0, XFMR["220/132_2x160"], 320, False, "Inaruwa 220/132", "xfmr",
        "Two 160-MVA units represented as one parallel equivalent; nominal ratio "
        "is absorbed by bus bases; off-nominal tap assumed 1.0 pu",
        tap=1.0, physical_units=2, parameter_status="source-derived + assumed",
        source_ref="[D2082] multiplicity/rating; 12% reactance and tap assumed")
    add(5, 11, r_ad, XFMR["assumed_220/132_100"] + x_ad, 100, True,
        "Amarpur-Dhungesanghu 132 (tie+xfmr)", "tie",
        "Prospective 132-kV tie plus a hypothetical 100-MVA 220/132-kV "
        "interface; no such interface rating/reactance is attributed to NEA",
        tap=1.0, length_km=19.13, physical_units=1,
        parameter_status="planned topology + assumed interface",
        source_ref="[D2082 project 8] planned route/length; all interface, conductor, rating, X% and tap values assumed")

    # Koshi Corridor 220 kV -- DOUBLE CIRCUIT (each circuit switchable) [D2082 KC1/KC4]
    add(1, 2, r_ib, x_ib, 600, True, "Basantapur-Inaruwa ckt-A", "line",
        length_km=57, parameter_status="in-service route + assumed parameters",
        source_ref="[D2082 KC1] route and aggregate corridor length; segment length, electrical parameters and rating assumed")
    add(1, 2, r_ib, x_ib, 600, True, "Basantapur-Inaruwa ckt-B", "line",
        length_km=57, parameter_status="planned circuit + assumed parameters",
        source_ref="[D2082 KC5] planned second circuit; segment length, electrical parameters and rating assumed")
    add(2, 3, r_bb, x_bb, 600, True, "Baneshwar-Basantapur ckt-A", "line",
        length_km=30, parameter_status="in-service route + assumed parameters",
        source_ref="[D2082 KC1] route and aggregate corridor length; segment length, electrical parameters and rating assumed")
    add(2, 3, r_bb, x_bb, 600, True, "Baneshwar-Basantapur ckt-B", "line",
        length_km=30, parameter_status="planned circuit + assumed parameters",
        source_ref="[D2082 KC5] planned second circuit; segment length, electrical parameters and rating assumed")
    add(3, 4, r_tb, x_tb, 600, True, "Tumlingtar-Baneshwar ckt-A", "line",
        length_km=20, parameter_status="in-service route + assumed parameters",
        source_ref="[D2082 KC1] route and aggregate corridor length; segment length, electrical parameters and rating assumed")
    add(3, 4, r_tb, x_tb, 600, True, "Tumlingtar-Baneshwar ckt-B", "line",
        length_km=20, parameter_status="planned circuit + assumed parameters",
        source_ref="[D2082 KC5] planned second circuit; segment length, electrical parameters and rating assumed")
    add(2, 5, r_db, x_db, 600, True, "Dhungesanghu-Basantapur ckt-A", "line",
        length_km=35, parameter_status="source-derived + assumed",
        source_ref="[D2082 KC3] topology/length; conductor and rating assumed")
    add(2, 5, r_db, x_db, 600, True, "Dhungesanghu-Basantapur ckt-B", "line",
        length_km=35, parameter_status="source-derived + assumed",
        source_ref="[D2082 KC3] topology/length; conductor and rating assumed")

    # 400 kV backbone (fixed)
    add(0, 6, r_dl, x_dl, 1600, False, "Inaruwa-Dhalkebar 400", "line",
        "Hetauda-Dhalkebar-Inaruwa 400 kV backbone [D2082]",
        length_km=84, parameter_status="source-derived + assumed",
        source_ref="[D2082] topology/approximate length; conductor and rating assumed")

    # Eastern 132 kV network (mix of fixed + tie switches -> real loops)
    add(1, 8, r_ik, x_ik, 160, True, "Inaruwa-Kusaha 132", "tie",
        length_km=22, source_ref="[MAP] topology; length, conductor and rating assumed")
    add(7, 8, r_dk, x_dk, 160, True, "Duhabi-Kusaha 132", "tie",
        length_km=28, parameter_status="source-derived + assumed",
        source_ref="[D2082 Package C] topology/length; conductor and rating assumed")
    add(7, 14, r_di, x_di, 160, False, "Duhabi-Itahari 132", "line",
        length_km=15, source_ref="[MAP] topology; length, conductor and rating assumed")
    add(14, 10, r_it, x_it, 160, False, "Itahari-Damak 132", "line",
        length_km=18, source_ref="[MAP] topology; length, conductor and rating assumed")
    add(10, 9, r_da, x_da, 160, False, "Damak-Anarmani 132", "line",
        length_km=45, source_ref="[D2082/MAP] topology; length, conductor and rating assumed")
    add(14, 15, r_idh, x_idh, 160, True, "Itahari-Dharan 132", "tie",
        length_km=16, source_ref="[MAP] topology; length, conductor and rating assumed")

    # Kabeli corridor 132 kV (Godak-Phidim-Amarpur) + tie to eastern 132 backbone
    add(11, 12, r_ap, x_ap, 160, True, "Amarpur-Phidim 132", "tie",
        length_km=58, source_ref="[D2082/MAP] topology; length, conductor and rating assumed")
    add(12, 13, r_pg, x_pg, 160, False, "Phidim-Godak 132", "line",
        length_km=55, source_ref="[D2082/MAP] topology; length, conductor and rating assumed")
    add(13, 10, r_gd, x_gd, 160, True, "Godak-Damak 132", "tie",
        "closes the Kabeli-to-eastern-network loop",
        length_km=60, source_ref="[MAP] topology; length, conductor and rating assumed")

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
              f"r={b.r_pu:.4f} x={b.x_pu:.4f} tap={b.tap_ratio_pu:.4f} "
              f"rating={b.rating_mva:.0f}MVA sw={int(b.switchable)} {b.kind}")
