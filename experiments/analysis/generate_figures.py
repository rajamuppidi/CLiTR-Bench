"""
Publication Figure Generator - CLiTR-Bench CMS125.

Generates four publication-quality figures (300 DPI, no overlapping text).

Figure 1 - Grouped bar chart: F1, Precision, Recall, AMR across all 6 conditions
Figure 2 - Stacked bar: Zero-shot error taxonomy (Qwen vs GPT-4o)
Figure 3 - CONSORT-style cohort derivation flowchart
Figure 4 - AMR and hallucination counts by prompt strategy

Output: experiments/analysis/figures/
Usage:  python3 experiments/analysis/generate_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR  = os.path.join(ANALYSIS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.3,
})

# ── Colors ────────────────────────────────────────────────────────────────────
C_GS   = "#2E86AB"   # guideline-supplied
C_ZSB  = "#E84855"   # zero-shot
METRIC_COLORS = {
    "F1":        "#2E86AB",
    "Precision": "#A23B72",
    "Recall":    "#F18F01",
    "AMR":       "#44BBA4",
}


# =============================================================================
# FIGURE 1 - Performance grouped bar chart (4 metrics x 6 conditions)
# =============================================================================
def figure1_grouped_bars():
    labels = [
        "GPT-4o\nGS", "Llama 70B\nGS", "Qwen 80B\nGS",
        "Llama 70B\nZSB", "Qwen 80B\nZSB", "GPT-4o\nZSB",
    ]
    data = {
        "F1":        [96.55, 87.66, 76.92, 47.02, 48.73, 49.68],
        "Precision": [97.90, 83.85, 64.52, 48.55, 33.25, 36.11],
        "Recall":    [95.24, 91.84, 95.24, 45.58, 91.16, 79.59],
        "AMR":       [98.00, 92.40, 82.60, 69.80, 43.60, 35.00],
    }
    f1_lo = [94.16, 83.44, 71.88, 39.60, 43.49, 44.06]
    f1_hi = [98.50, 91.30, 81.44, 53.87, 53.70, 54.95]

    n  = len(labels)
    m  = len(data)
    bw = 0.17
    x  = np.arange(n)
    offsets = np.linspace(-(m - 1) / 2 * bw, (m - 1) / 2 * bw, m)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(top=0.78)   # leave room for legend above plot

    bar_handles = []
    for i, (metric, vals) in enumerate(data.items()):
        bars = ax.bar(x + offsets[i], vals, width=bw,
                      color=METRIC_COLORS[metric], alpha=0.85,
                      label=metric, edgecolor="white", linewidth=0.4)
        bar_handles.append(bars)

    # F1 CI error bars only
    fi = list(data.keys()).index("F1")
    f1_vals = data["F1"]
    ax.errorbar(
        x + offsets[fi], f1_vals,
        yerr=[[v - lo for v, lo in zip(f1_vals, f1_lo)],
              [hi - v for v, hi in zip(f1_vals, f1_hi)]],
        fmt="none", color="#1a1a2e", capsize=3, capthick=1.2, linewidth=1.3, zorder=5,
        label="F1 95% CI"
    )

    # Divider between guideline and zero-shot sections
    ax.axvline(2.5, color="#aaa", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    # Section labels placed on the x-axis area below the bars
    ax.text(1.0, -14, "Guideline-Supplied", ha="center", fontsize=10,
            color=C_GS, fontweight="bold", clip_on=False)
    ax.text(4.0, -14, "Zero-Shot Base", ha="center", fontsize=10,
            color=C_ZSB, fontweight="bold", clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Place legend entirely above the axes in figure space
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.05),
        ncol=5,
        fontsize=10,
        framealpha=0.95,
        columnspacing=1.2,
        handlelength=1.4,
        borderpad=0.6,
    )

    ax.set_title(
        "Figure 1. CLiTR-Bench CMS125 - Model Performance Across All Conditions\n"
        "(n=500 patients; error bars on F1 represent 95% bootstrap CI, B=10,000)",
        fontsize=11, pad=50   # large pad to sit above legend
    )

    out = os.path.join(FIGURES_DIR, "figure1_performance_bars.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")



# =============================================================================
# FIGURE 2 - Stacked error taxonomy
# =============================================================================
def figure2_error_taxonomy():
    models = ["Qwen 3 80B\nZero-Shot\n(n=282 errors)", "GPT-4o\nZero-Shot\n(n=237 errors)"]
    wrong   = [92.9, 51.1]
    outside = [2.5,  36.3]
    noev    = [4.6,  12.7]

    x  = np.arange(len(models))
    bw = 0.40
    c1, c2, c3 = "#E84855", "#F18F01", "#AAAAAA"

    fig, ax = plt.subplots(figsize=(8, 8))
    # Leave room at bottom for x-axis labels and legend
    fig.subplots_adjust(bottom=0.30, top=0.87)

    ax.bar(x, wrong,   width=bw, color=c1, alpha=0.88, label="WRONG_CONCLUSION")
    ax.bar(x, outside, width=bw, color=c2, alpha=0.88, label="OUTSIDE_WINDOW",
           bottom=wrong)
    ax.bar(x, noev,    width=bw, color=c3, alpha=0.88, label="NO_EVIDENCE_CITED",
           bottom=[a + b for a, b in zip(wrong, outside)])

    # Segment labels: only draw if segment is at least 6% tall
    for i in range(len(models)):
        mid1 = wrong[i] / 2
        ax.text(x[i], mid1, f"{wrong[i]:.1f}%",
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="white")

        if outside[i] >= 6:
            mid2 = wrong[i] + outside[i] / 2
            ax.text(x[i], mid2, f"{outside[i]:.1f}%",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="white")

        if noev[i] >= 6:
            mid3 = wrong[i] + outside[i] + noev[i] / 2
            ax.text(x[i], mid3, f"{noev[i]:.1f}%",
                    ha="center", va="center", fontsize=11, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Percentage of Non-Auditable Cases (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_title(
        "Figure 2. Zero-Shot Hallucination Error Type Distribution",
        fontsize=12, pad=10
    )

    # Legend placed below x-axis via figure coordinates
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,
        fontsize=11,
        framealpha=0.95,
        handlelength=1.6,
        borderpad=0.7,
    )

    out = os.path.join(FIGURES_DIR, "figure2_error_taxonomy.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")



# =============================================================================
# FIGURE 3 - CONSORT flowchart (pure matplotlib, no overlapping text)
# =============================================================================
def figure3_consort():
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def draw_box(cx, cy, w, h, lines, bg="#DBEAFE", border="#2E86AB", fs=9.5):
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=bg, edgecolor=border, linewidth=1.5
        )
        ax.add_patch(rect)
        text = "\n".join(lines)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, multialignment="center", color="#1a1a2e",
                linespacing=1.5)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#2E86AB",
                            lw=1.6, mutation_scale=14)
        )

    def draw_excl(cx, cy, lines, fs=8.5):
        text = "\n".join(lines)
        ax.text(cx, cy, text, ha="left", va="center", fontsize=fs,
                linespacing=1.4, color="#7f1d1d",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEE2E2",
                          edgecolor="#EF4444", linewidth=1.1))

    # Row y-positions
    y0, y1, y2, y3, y4, y5, y6, y7 = 13.2, 11.6, 10.1, 8.6, 7.2, 5.5, 3.8, 2.0

    draw_box(5, y0, 8.5, 0.9, ["Synthea simulation: 25,000 California patients"], bg="#EFF6FF")
    draw_arrow(5, y0 - 0.45, 5, y1 + 0.6)

    draw_box(5, y1, 8.5, 1.0,
             ["Initial population filter: Female, aged 52-74 as of 2025-12-31",
              "Eligible: 3,002 patients"])
    draw_excl(5.6, y1 - 0.8, ["Excluded: 21,998 (wrong sex, age, or both)"])
    draw_arrow(5, y1 - 0.5, 5, y2 + 0.5)

    draw_box(5, y2, 8.5, 1.0,
             ["Denominator filter: >=1 qualifying 2025 clinical encounter",
              "Eligible: 2,987 patients"])
    draw_excl(5.6, y2 - 0.8, ["Excluded: 15 (no qualifying encounter in 2025)"])
    draw_arrow(5, y2 - 0.5, 5, y3 + 0.5)

    draw_box(5, y3, 8.5, 1.0,
             ["Exclusion criteria: bilateral mastectomy history",
              "Remaining: 2,987 patients"])
    draw_excl(5.6, y3 - 0.8, ["Excluded: 0 (no mastectomy events in Synthea)"])
    draw_arrow(5, y3 - 0.5, 5, y4 + 0.6)

    draw_box(5, y4, 8.5, 1.1,
             ["Gold truth evaluation (deterministic engine)",
              "Compliant (mammogram in 27-month window): 147 (4.9%)",
              "Non-compliant: 2,840 (95.1%)"], bg="#F0FDF4")

    # Diverge
    draw_arrow(3.0, y4 - 0.55, 2.2, y5 + 0.45)
    draw_arrow(7.0, y4 - 0.55, 7.8, y5 + 0.45)

    draw_box(2.2, y5, 3.8, 0.8,
             ["All 147 compliant", "patients included"], bg="#F0FDF4")
    draw_box(7.8, y5, 3.8, 0.8,
             ["352 non-compliant patients", "random sample (seed=99)"], bg="#FFF7ED")

    # Merge
    draw_arrow(2.2, y5 - 0.4, 3.8, y6 + 0.45)
    draw_arrow(7.8, y5 - 0.4, 6.2, y6 + 0.45)

    draw_box(5, y6, 8.8, 1.2,
             ["Publication cohort: n=499 patients",
              "Compliant: 147 | Non-compliant: 352",
              "Compliance rate: 29.4% (natural Synthea rate: 4.9%)"], bg="#EFF6FF")
    draw_arrow(5, y6 - 0.6, 5, y7 + 0.5)

    draw_box(5, y7, 8.8, 0.9,
             ["LLM inference: 6 conditions x ~500 calls = 3,000 total API calls",
              "GPT-4o GS  |  Llama 70B GS  |  Qwen 80B GS  |  GPT-4o ZSB  |  Llama ZSB  |  Qwen ZSB"],
             bg="#F5F3FF", fs=9.0)

    ax.set_title(
        "Figure 3. Patient Cohort Derivation (CMS125 HEDIS 2025 BCS-E)\n"
        "Synthea Synthetic Patient Population, Index Date: 2025-12-31",
        fontsize=11, pad=12, fontweight="bold"
    )

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure3_consort_flowchart.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# FIGURE 4 - AMR and Hallucination counts side by side
# =============================================================================
def figure4_amr_hallucinations():
    models  = ["GPT-4o", "Llama 3.3 70B", "Qwen 3 80B"]
    amr_gs  = [98.0, 92.4, 82.6]
    amr_zs  = [35.0, 69.8, 43.6]
    hall_gs = [10,   38,   89]
    hall_zs = [325,  196,  282]

    x  = np.arange(len(models))
    bw = 0.32
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: AMR
    b1 = ax1.bar(x - bw / 2, amr_gs, width=bw, color=C_GS,  alpha=0.87,
                 label="Guideline-Supplied")
    b2 = ax1.bar(x + bw / 2, amr_zs, width=bw, color=C_ZSB, alpha=0.87,
                 label="Zero-Shot Base")
    for bar, v in list(zip(b1, amr_gs)) + list(zip(b2, amr_zs)):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.2,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel("Auditability Match Rate (%)", fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax1.set_axisbelow(True)
    ax1.set_title("Auditability Match Rate (AMR)\nby Model and Prompt Strategy", fontsize=11)
    ax1.legend(fontsize=10, framealpha=0.9)

    # Right: Hallucinations
    b3 = ax2.bar(x - bw / 2, hall_gs, width=bw, color=C_GS,  alpha=0.87,
                 label="Guideline-Supplied")
    b4 = ax2.bar(x + bw / 2, hall_zs, width=bw, color=C_ZSB, alpha=0.87,
                 label="Zero-Shot Base")
    for bar, v in list(zip(b3, hall_gs)) + list(zip(b4, hall_zs)):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 4,
                 str(v), ha="center", va="bottom", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_ylabel("Hallucinated Evidence Citations (count)", fontsize=11)
    ax2.set_ylim(0, 380)
    ax2.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.set_title("Hallucinated Evidence Citations\nby Model and Prompt Strategy", fontsize=11)
    ax2.legend(fontsize=10, framealpha=0.9)

    fig.suptitle(
        "Figure 4. Clinical Auditability: AMR and Hallucination Count by Model and Prompt Strategy",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure4_amr_hallucinations.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating CLiTR-Bench publication figures...")
    figure1_grouped_bars()
    figure2_error_taxonomy()
    figure3_consort()
    figure4_amr_hallucinations()
    print(f"\nAll figures saved to: {FIGURES_DIR}")
