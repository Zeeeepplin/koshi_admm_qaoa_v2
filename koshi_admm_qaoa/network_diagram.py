import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from network_data import build_full_network


def draw_single_line_diagram(net, save_path="eastern_nepal_sld_compact.png"):
    """Generates a compact, professional single-line diagram (SLD) that fits

    in a single view screen/window.
    """
    G = nx.MultiGraph()

    # Add Nodes
    for b in net.buses:
        G.add_node(
            b.idx,
            name=b.name,
            kv=b.kv,
            is_slack=b.is_slack,
            is_gen=(b.p_load_mw < 0),
            p_load=b.p_load_mw,
        )

    # Add Branches
    for br in net.branches:
        G.add_edge(
            br.frm,
            br.to,
            key=br.idx,
            name=br.name,
            switchable=br.switchable,
            kind=br.kind,
        )

    # Voltage Color Palette
    color_map = {
        400.0: "#D62728",  # Red (400 kV)
        220.0: "#1F77B4",  # Blue (220 kV)
        132.0: "#2CA02C",  # Green (132 kV)
    }

  
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    plt.title(
        "16-Bus Eastern Nepal Transmission Sub-System (SLD)\n"
        "Adapted from NEA Transmission & PMD Directorate Year Book FY 2024/25",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )


    pos = {
        0: (0.0, 0.0),  # Inaruwa 400 (Slack Hub)
        1: (0.0, 0.9),  # Inaruwa 220
        2: (0.0, 2.2),  # Basantapur 220
        3: (-0.9, 3.2),  # Baneshwar 220
        4: (-1.8, 4.0),  # Tumlingtar 220 (Gen)
        5: (1.2, 3.2),  # Dhungesanghu 220 (Gen)
        6: (-1.8, 0.0),  # Dhalkebar 400 (Backbone)
        7: (0.9, 0.5),  # Duhabi 132
        8: (0.6, -0.7),  # Kusaha 132
        9: (4.6, -0.4),  # Anarmani 132
        10: (3.5, -0.4),  # Damak 132
        11: (2.3, 3.2),  # Amarpur 132 (Gen)
        12: (3.2, 2.2),  # Phidim 132
        13: (4.0, 1.1),  # Godak 132
        14: (2.1, 0.5),  # Itahari 132
        15: (2.1, 1.6),  # Dharan 132
    }


    for bus in net.buses:
        n_color = color_map.get(bus.kv, "gray")
        n_shape = "s" if bus.is_slack else "o"
        n_size = 450 if bus.is_slack else (350 if bus.p_load_mw < 0 else 250)

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[bus.idx],
            node_color=n_color,
            node_shape=n_shape,
            node_size=n_size,
            edgecolors="black",
            linewidths=1.2,
            ax=ax,
        )


    labels = {}
    for b in net.buses:
        short_name = b.name.split("_")[0]
        if b.is_slack:
            labels[b.idx] = f"{b.idx}:{short_name}\n(Slack)"
        elif b.p_load_mw < 0:
            labels[b.idx] = f"{b.idx}:{short_name}\n({abs(b.p_load_mw):.0f}MW)"
        else:
            labels[b.idx] = f"{b.idx}:{short_name}"

    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=6.5, font_weight="bold", ax=ax
    )


    edge_counts = {}
    for u, v, k in G.edges(keys=True):
        pair = tuple(sorted((u, v)))
        edge_counts[pair] = edge_counts.get(pair, 0) + 1

    edge_tracker = {}
    for u, v, k, data in G.edges(data=True, keys=True):
        pair = tuple(sorted((u, v)))
        total_parallel = edge_counts[pair]
        curr_idx = edge_tracker.get(pair, 0)
        edge_tracker[pair] = curr_idx + 1

    
        if total_parallel > 1:
            rad = 0.12 * (curr_idx - (total_parallel - 1) / 2.0)
            conn_style = f"arc3,rad={rad}"
        else:
            conn_style = "arc3,rad=0.0"

        line_style = "dashed" if data["switchable"] else "solid"
        edge_color = "#333333" if not data["switchable"] else "#666666"
        width = 1.4 if not data["switchable"] else 1.0

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            style=line_style,
            edge_color=edge_color,
            width=width,
            connectionstyle=conn_style,
            ax=ax,
        )

    # Compact Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="400 kV (Slack)",
            markerfacecolor="#D62728",
            markeredgecolor="k",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="220 kV",
            markerfacecolor="#1F77B4",
            markeredgecolor="k",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="132 kV",
            markerfacecolor="#2CA02C",
            markeredgecolor="k",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=1.5,
            linestyle="-",
            label="Fixed Edge",
        ),
        Line2D(
            [0],
            [0],
            color="#666666",
            lw=1.2,
            linestyle="--",
            label="Switchable Edge",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=7,
        frameon=True,
        labelspacing=0.3,
        handletextpad=0.4,
    )

    ax.margins(0.08)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    net = build_full_network()
    draw_single_line_diagram(net)