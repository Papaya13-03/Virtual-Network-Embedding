#!/usr/bin/env python3
"""Generate diagram images for the VNE presentation."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

# === Color Palette (matching PPTX) ===
BG = "#1B1B2F"
CARD = "#24243E"
BLUE = "#4E9AF5"
CYAN = "#56CCF2"
GREEN = "#48D1A5"
ORANGE = "#FF9F43"
PURPLE = "#A27AF0"
RED = "#FF6B6B"
WHITE = "#F0F0F0"
GRAY = "#B0B0C0"


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none', transparent=False)
    plt.close(fig)
    print(f"  Saved {name}")


def box(ax, cx, cy, w, h, title, body_lines, color=BLUE, title_size=12, body_size=10):
    """Draw a rounded box with title + body text, properly spaced."""
    bx = cx - w / 2
    by = cy - h / 2
    patch = FancyBboxPatch((bx, by), w, h, boxstyle="round,pad=0.06",
                           facecolor=CARD, edgecolor=color, linewidth=2, zorder=2)
    ax.add_patch(patch)

    line_h = h / (1 + len(body_lines) + 0.6)  # divide box height among lines
    top = cy + h / 2 - line_h * 0.7

    ax.text(cx, top, title, ha='center', va='center',
            fontsize=title_size, color=color, fontweight='bold', zorder=3)

    for i, line in enumerate(body_lines):
        yy = top - (i + 1) * line_h
        ax.text(cx, yy, line, ha='center', va='center',
                fontsize=body_size, color=WHITE, zorder=3)


def sbox(ax, cx, cy, w, h, text, color=BLUE, fontsize=12, bold=True):
    """Simple box with single centered text."""
    bx = cx - w / 2
    by = cy - h / 2
    patch = FancyBboxPatch((bx, by), w, h, boxstyle="round,pad=0.06",
                           facecolor=CARD, edgecolor=color, linewidth=2, zorder=2)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color=color if bold else WHITE,
            fontweight='bold' if bold else 'normal', zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)


def diamond(ax, cx, cy, rx, ry, text, color=ORANGE, fontsize=11):
    d = plt.Polygon([
        [cx, cy + ry], [cx + rx, cy], [cx, cy - ry], [cx - rx, cy]
    ], fc=CARD, ec=color, lw=2.5, zorder=3)
    ax.add_patch(d)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight='bold', zorder=4)


def node(ax, x, y, label, color=BLUE, r=0.3, fontsize=12):
    c = plt.Circle((x, y), r, fc=CARD, ec=color, lw=2, zorder=4)
    ax.add_patch(c)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight='bold', zorder=5)


def link(ax, x1, y1, x2, y2, color=BLUE, lw=1.5, alpha=0.4):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)


def setup_ax(fig_w, fig_h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


# ============================================================
# 1. NETWORK VIRTUALIZATION
# ============================================================
def diagram_network_virtualization():
    fig, ax = setup_ax(12, 6.5, (-0.5, 12.5), (-0.5, 6.8))

    # VN1
    ax.text(2.5, 6.3, "VN Request 1", ha='center', fontsize=14, color=CYAN, fontweight='bold')
    vn1 = [(1.5, 5.3), (3.5, 5.3), (2.5, 4.3)]
    for p, l in zip(vn1, "ABC"):
        node(ax, *p, l, CYAN, 0.3, 13)
    for a, b in [(0,1),(0,2),(1,2)]:
        link(ax, *vn1[a], *vn1[b], CYAN, 2, 0.6)

    # VN2
    ax.text(9, 6.3, "VN Request 2", ha='center', fontsize=14, color=GREEN, fontweight='bold')
    vn2 = [(7.5, 5.3), (9.5, 5.3), (7.5, 4.3), (9.5, 4.3)]
    for p, l in zip(vn2, "XYZW"):
        node(ax, *p, l, GREEN, 0.3, 13)
    for a, b in [(0,1),(0,2),(1,3),(2,3)]:
        link(ax, *vn2[a], *vn2[b], GREEN, 2, 0.6)

    # Arrow
    ax.annotate('', xy=(5.75, 3.3), xytext=(5.75, 4.0),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=3))
    ax.text(5.75, 3.65, "Embed onto", ha='center', fontsize=13, color=ORANGE, fontweight='bold')

    # Substrate
    ax.text(5.75, 3.0, "Physical Substrate Network", ha='center', fontsize=14, color=BLUE, fontweight='bold')
    sn = [(2,2.2),(4,2.2),(6,2.2),(8,2.2),(10,2.2),
          (2,0.8),(4,0.8),(6,0.8),(8,0.8),(10,0.8)]
    for i, (x, y) in enumerate(sn):
        node(ax, x, y, str(i+1), BLUE, 0.28, 11)
    sn_links = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),
                (0,5),(1,6),(2,7),(3,8),(4,9)]
    for a, b in sn_links:
        link(ax, *sn[a], *sn[b], BLUE, 1.5, 0.3)

    # Highlight mapped
    for i in [0, 2, 5]:
        plt.gca().add_patch(plt.Circle(sn[i], 0.38, fc='none', ec=CYAN, lw=2.5, ls='--', zorder=6))
    for i in [3, 4, 7, 8]:
        plt.gca().add_patch(plt.Circle(sn[i], 0.38, fc='none', ec=GREEN, lw=2.5, ls='--', zorder=6))

    save(fig, "network_virtualization.png")


# ============================================================
# 2. TWO-PHASE PIPELINE
# ============================================================
def diagram_two_phase_pipeline():
    fig, ax = setup_ax(10, 9, (0, 10), (0, 9))

    sbox(ax, 5, 8.2, 5.5, 0.8, "VN Request (Gv) arrives", ORANGE, 14)
    arrow(ax, 5, 7.8, 5, 7.2)

    box(ax, 5, 6.2, 6.5, 1.6, "Phase 1: NODE MAPPING",
        ["Greedy / PSO / Q-Learning / DQN",
         "selects substrate node for each virtual node"],
        BLUE, 14, 11)
    arrow(ax, 5, 5.4, 5, 4.8)

    box(ax, 5, 3.8, 6.5, 1.6, "Phase 2: LINK MAPPING",
        ["Kruskal MST / Dijkstra / Floyd",
         "Single-path or Multi-path (up to 5)"],
        CYAN, 14, 11)
    arrow(ax, 5, 3.0, 5, 2.4)

    diamond(ax, 5, 1.9, 1.2, 0.45, "Success?", ORANGE, 13)

    # Yes
    arrow(ax, 6.2, 1.9, 7.5, 1.9, GREEN, 2.5)
    box(ax, 8.8, 1.9, 2.2, 0.9, "Commit",
        ["Deduct CPU & BW"], GREEN, 12, 10)

    # No
    arrow(ax, 3.8, 1.9, 2.5, 1.9, RED, 2.5)
    box(ax, 1.3, 1.9, 2.2, 0.9, "Rollback",
        ["Restore ALL resources"], RED, 12, 10)

    save(fig, "two_phase_pipeline.png")


# ============================================================
# 3. SYSTEM ARCHITECTURE
# ============================================================
def diagram_system_architecture():
    fig, ax = setup_ax(12, 7, (-0.5, 12.5), (-0.5, 7.5))

    ax.text(6, 7.1, "Centralized Hierarchical Multi-Domain Architecture",
            ha='center', fontsize=15, color=WHITE, fontweight='bold')

    box(ax, 6, 5.8, 9, 1.8, "Global Controller",
        ["Receives VN Requests",
         "Runs optimization (Greedy / PSO / RL)",
         "Coordinates cross-domain routing",
         "Atomic commit / rollback"],
        BLUE, 14, 11)

    for x in [2.5, 6, 9.5]:
        arrow(ax, x, 4.9, x, 4.2)

    for x, col, label in [(2.5, GREEN, "Domain 1"), (6, CYAN, "Domain 2"), (9.5, PURPLE, "Domain 3")]:
        box(ax, x, 2.8, 3.2, 2.4, f"Local Controller — {label}",
            ["30 nodes  •  2 boundary",
             "Resource tracking",
             "Candidate selection",
             "Intra-domain paths (cached)"],
            col, 11, 10)

    # Inter-domain links
    ax.annotate('', xy=(4.3, 2.8), xytext=(3.9, 2.8),
                arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
    ax.annotate('', xy=(8.1, 2.8), xytext=(7.7, 2.8),
                arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))

    save(fig, "system_architecture.png")


# ============================================================
# 4. ALGORITHM EVOLUTION
# ============================================================
def diagram_algorithm_evolution():
    fig, ax = setup_ax(13, 7.5, (-0.5, 13), (-1.5, 7.5))

    ax.text(6.25, 7.0, "Algorithm Evolution: From Greedy to Swarm RL Hybrids",
            ha='center', fontsize=15, color=WHITE, fontweight='bold')

    # Legend
    for i, (label, col) in enumerate([
        ("Heuristic", GRAY), ("Metaheuristic", BLUE), ("Tabular RL", CYAN),
        ("Deep RL", PURPLE), ("Hybrid", ORANGE), ("Full Hybrid", GREEN)
    ]):
        x = 0.3 + i * 2.15
        patch = FancyBboxPatch((x, 6.3), 0.3, 0.2, boxstyle="round,pad=0.02",
                               facecolor=col, edgecolor='none', alpha=0.8)
        ax.add_patch(patch)
        ax.text(x + 0.45, 6.4, label, fontsize=9, color=WHITE, va='center')

    # Nodes
    sbox(ax, 6.25, 5.3, 4.5, 0.8, "MC-VNM  |  Greedy + Kruskal MST", GRAY, 12)

    arrow(ax, 4.5, 4.9, 3, 4.3)
    arrow(ax, 8, 4.9, 9.5, 4.3)

    sbox(ax, 3, 3.7, 4.5, 0.8, "MP-VNE  |  PSO + Mutation + Multi-path", BLUE, 11)
    sbox(ax, 9.5, 3.7, 4.5, 0.8, "MPQ-VNE  |  Q-Learning + Dijkstra", CYAN, 11)

    arrow(ax, 9.5, 3.3, 7.5, 2.3)

    sbox(ax, 7.5, 1.7, 5.5, 0.8, "SRL-VNE  |  DQN + PSO Swarm + Dijkstra", PURPLE, 11)

    arrow(ax, 3, 3.3, 3, -0.3)
    arrow(ax, 5.5, 1.3, 3.5, -0.3)
    arrow(ax, 9, 1.3, 10, -0.3)

    sbox(ax, 3, -0.8, 5, 0.8, "MP-DQN-VNE  |  PSO + DQN + Multi-path", ORANGE, 11)
    sbox(ax, 10, -0.8, 5, 0.8, "SRL-MP-VNE  |  Swarm RL + PSO + Multi-path", GREEN, 10)

    save(fig, "algorithm_evolution.png")


# ============================================================
# 5. PSO SWARM
# ============================================================
def diagram_pso_swarm():
    fig, ax = setup_ax(12, 7, (-0.5, 12.5), (-0.5, 7))

    ax.text(6, 6.6, "Swarm RL: 4 DQN Agents with PSO Knowledge Sharing",
            ha='center', fontsize=15, color=WHITE, fontweight='bold')

    colors = [BLUE, CYAN, GREEN, PURPLE]
    xs = [1.8, 4.4, 7.0, 9.6]
    for i, (x, col) in enumerate(zip(xs, colors)):
        box(ax, x, 5.0, 2.2, 1.6, f"Agent {i+1}",
            ["DQN: 6→64→64→out", "ε-greedy explore"],
            col, 12, 10)

    # Arrows to center
    for x in xs:
        arrow(ax, x, 4.2, 5.7, 3.3, GRAY, 1.5)

    box(ax, 5.7, 2.3, 7, 1.6, "PSO Information Sharing",
        ["Track personal best (pBest) per agent",
         "Track global best (gBest) across swarm",
         "Q += β(Q_pBest − Q) + δ(Q_gBest − Q)"],
        ORANGE, 13, 11)

    arrow(ax, 5.7, 1.5, 5.7, 0.8)

    sbox(ax, 5.7, 0.3, 5, 0.7, "Best Policy → Node Selection", GREEN, 13)

    save(fig, "pso_swarm.png")


# ============================================================
# 6. MP-DQN-VNE PIPELINE
# ============================================================
def diagram_mp_dqn_pipeline():
    fig, ax = setup_ax(12, 8, (-0.5, 12.5), (-0.5, 8))

    ax.text(6, 7.6, "MP-DQN-VNE: PSO Guided by DQN Fitness",
            ha='center', fontsize=15, color=WHITE, fontweight='bold')

    box(ax, 3.5, 5.8, 5.5, 3, "PSO Optimization",
        ["15 particles × 10 iterations", "",
         "For each particle:",
         "1. Compute node_cost",
         "2. Extract state features",
         "3. Query DQN → max_q",
         "4. penalty = −10 × max_q",
         "5. fitness = cost + penalty"],
        BLUE, 13, 10)

    box(ax, 9.5, 6.2, 4.5, 2, "Global Best DQN",
        ["4 swarm agents",
         "State: [v_cpu, s_cpu, s_bw,",
         "  s_degree, progress, bias]",
         "Higher Q → lower fitness"],
        PURPLE, 12, 10)

    # Arrows between PSO and DQN
    ax.annotate('query', xy=(7.5, 6.5), xytext=(6.25, 6.5),
                fontsize=10, color=ORANGE, fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))
    ax.annotate('Q-value', xy=(6.25, 5.8), xytext=(7.5, 5.8),
                fontsize=10, color=CYAN, fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=2))

    arrow(ax, 3.5, 4.3, 3.5, 3.5)
    sbox(ax, 3.5, 3.0, 5.5, 0.7, "Best Node Mapping (gbest)", GREEN, 13)

    arrow(ax, 3.5, 2.65, 3.5, 2.0)
    sbox(ax, 3.5, 1.5, 5.5, 0.7, "Multi-path Link Allocation (≤ 5 paths)", CYAN, 12)

    # Reward loop
    arrow(ax, 6.25, 1.5, 9.5, 2.5)
    box(ax, 9.5, 3.3, 4.5, 1.4, "Training",
        ["R = 100 × Revenue / Cost",
         "All 4 agents learn",
         "Update pBest, gBest"],
        ORANGE, 12, 10)

    save(fig, "mp_dqn_pipeline.png")


# ============================================================
# 7. MULTI-PATH ALLOCATION
# ============================================================
def diagram_multipath():
    fig, ax = setup_ax(12, 6, (-0.5, 12.5), (-0.8, 5.8))

    ax.text(5.5, 5.4, "Multi-Path Link Allocation Example",
            ha='center', fontsize=15, color=WHITE, fontweight='bold')
    ax.text(5.5, 4.9, "Virtual link A→B  |  Demand = 80 BW",
            ha='center', fontsize=12, color=ORANGE)

    # Network
    nodes_pos = {'A': (1, 2.5), '1': (3.5, 3.8), '2': (3.5, 1.2),
                 '3': (6, 3.8), '4': (6, 1.2), 'B': (8.5, 2.5)}

    all_links = [('A','1'),('A','2'),('1','3'),('2','4'),('3','B'),('4','B'),('1','4'),('2','3')]
    for a, b in all_links:
        x1, y1 = nodes_pos[a]
        x2, y2 = nodes_pos[b]
        link(ax, x1, y1, x2, y2, GRAY, 1, 0.3)

    # Path 1: A→1→3→B
    for a, b in [('A','1'),('1','3'),('3','B')]:
        x1, y1 = nodes_pos[a]
        x2, y2 = nodes_pos[b]
        link(ax, x1, y1, x2, y2, CYAN, 3.5, 0.8)

    # Path 2: A→2→4→B
    for a, b in [('A','2'),('2','4'),('4','B')]:
        x1, y1 = nodes_pos[a]
        x2, y2 = nodes_pos[b]
        link(ax, x1, y1, x2, y2, GREEN, 3.5, 0.8)

    for name, (x, y) in nodes_pos.items():
        is_end = name in ('A', 'B')
        node(ax, x, y, name, ORANGE if is_end else BLUE, 0.35 if is_end else 0.28, 13)

    # Legend cards
    box(ax, 10.5, 3.8, 2.5, 1.2, "Path 1", ["A→1→3→B", "Alloc: 50 BW"], CYAN, 12, 11)
    box(ax, 10.5, 1.8, 2.5, 1.2, "Path 2", ["A→2→4→B", "Alloc: 30 BW"], GREEN, 12, 11)

    # Result
    sbox(ax, 5.5, -0.2, 9, 0.7, "Total: 50 + 30 = 80 BW ✓  Demand satisfied with 2 paths", ORANGE, 13)

    save(fig, "multipath_allocation.png")


# ============================================================
# 8. SRL-MP-VNE PIPELINE
# ============================================================
def diagram_srl_mp_pipeline():
    fig, ax = setup_ax(13, 6.5, (-0.5, 13), (-0.5, 6.5))

    ax.text(6.25, 6.1, "SRL-MP-VNE: Complete Pipeline",
            ha='center', fontsize=15, color=WHITE, fontweight='bold')

    # Top row
    box(ax, 1.5, 4.5, 2.5, 1.4, "1. VNR",
        ["Find candidates"], ORANGE, 12, 10)
    arrow(ax, 2.75, 4.5, 3.5, 4.5, GRAY)

    box(ax, 4.8, 4.5, 2.5, 1.4, "2. PSO",
        ["10p × 50 iterations", "DQN-guided fitness"], BLUE, 12, 10)
    arrow(ax, 6.05, 4.5, 6.8, 4.5, GRAY)

    box(ax, 8.1, 4.5, 2.5, 1.4, "3. Multi-Path",
        ["Link allocation", "≤ 5 paths"], CYAN, 12, 10)
    arrow(ax, 9.35, 4.5, 10.1, 4.5, GRAY)

    box(ax, 11.3, 4.5, 2, 1.4, "4. Reward",
        ["+100 / -50"], GREEN, 12, 11)

    # Bottom row
    arrow(ax, 11.3, 3.8, 11.3, 2.8)

    box(ax, 10, 2, 2.8, 1.2, "5. Train",
        ["All 4 DQN agents", "Batch from memory"], PURPLE, 12, 10)
    arrow(ax, 8.6, 2, 7.8, 2, GRAY)

    box(ax, 6.5, 2, 2.5, 1.2, "6. Update",
        ["pBest / gBest", "PSO sharing"], ORANGE, 12, 10)

    # Loop back
    arrow(ax, 5.25, 2, 1.5, 2, GRAY, 1.5)
    arrow(ax, 1.5, 2, 1.5, 3.8, GRAY, 1.5)
    ax.text(3.5, 1.5, "← Loop for next VNR", fontsize=11, color=GRAY, ha='center')

    save(fig, "srl_mp_pipeline.png")


# ============================================================
# RUN ALL
# ============================================================
print("Generating diagrams...")
diagram_network_virtualization()
diagram_two_phase_pipeline()
diagram_system_architecture()
diagram_algorithm_evolution()
diagram_pso_swarm()
diagram_mp_dqn_pipeline()
diagram_multipath()
diagram_srl_mp_pipeline()
print(f"All diagrams saved to {OUT_DIR}")
