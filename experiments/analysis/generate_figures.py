"""
Publication Figure Generator — CLiTR-Bench CMS125.

Generates three publication-quality figures:
  Figure 1 — Grouped bar chart: F1, Precision, Recall, AMR across all 6 conditions
  Figure 2 — Stacked bar: Zero-shot error taxonomy (Qwen vs GPT-4o)
  Figure 3 — CONSORT-style cohort derivation flowchart

Output: experiments/analysis/figures/
Usage:  python3 experiments/analysis/generate_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR  = os.path.join(ANALYSIS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# ── Data ──────────────────────────────────────────────────────────────────────
CONDITIONS = [
    "GPT-4o\nGuideline",
    "Llama 3.3 70B\nGuideline",
    "Qwen 3 80B\nGuideline",
    "Llama 3.3 70B\nZero-Shot",
    "Qwen 3 80B\nZero-Shot",
    "GPT-4o\nZero-Shot",
]

METRICS = {
    "F1":        [96.55, 87.66, 76.92, 47.02, 48.73, 49.68],
    "Precision": [97.90, 83.85, 64.52, 48.55, 33.25, 36.11],
    "Recall":    [95.24, 91.84, 95.24, 45.58, 91.16, 79.59],
    "AMR":       [98.00, 92.40, 82.60, 69.80, 43.60, 35.00],
}

CI_F1_LOWER = [94.16, 83.44, 71.88, 39.60, 43.49, 44.06]
CI_F1_UPPER = [98.50, 91.30, 81.44, 53.87, 53.70, 54.95]

COLORS = {
    "F1":        "#2E86AB",
    "Precision": "#A23B72",
    "Recall":    "#F18F01",
    "AMR":       "#44BBA4",
}

GUIDELINE_COLOR = "#2E86AB"
ZEROSHOT_COLOR  = "#E84855"


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Grouped bar chart: all metrics across all 6 conditions
# ═══════════════════════════════════════════════════════════════════════════════
def figure1_grouped_bars():
    n_conds  = len(CONDITIONS)
    n_metrics = len(METRICS)
    bar_w    = 0.18
    x        = np.arange(n_conds)

    fig, ax = plt.subplots(figsize=(13, 6))

    offsets = np.linspace(-(n_metrics - 1) / 2 * bar_w, (n_metrics - 1) / 2 * bar_w, n_metrics)
    for i, (metric, values) in enumerate(METRICS.items()):
        bars = ax.bar(x + offsets[i], values, width=bar_w, label=metric,
                      color=COLORS[metric], alpha=0.88, edgecolor="white", linewidth=0.5)
        # Value labels on top of each bar
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333333")

    # Add F1 95% CI error bars
    f1_idx = list(METRICS.keys()).index("F1")
    f1_offset = offsets[f1_idx]
    f1_vals   = METRICS["F1"]
    f1_lower  = [v - l for v, l in zip(f1_vals, CI_F1_LOWER)]
    f1_upper  = [u - v for v, u in zip(f1_vals, CI_F1_UPPER)]
    ax.errorbar(x + f1_offset, f1_vals, yerr=[f1_lower, f1_upper],
                fmt="none", color="#1a1a2e", capsize=3, capthick=1.2, linewidth=1.2,
                label="F1 95% CI")

    # Divider between guideline and zero-shot conditions
    ax.axvline(x=2.5, color="#999999", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(0.95, 104, "Guideline-Supplied", ha="center", fontsize=9,
            color=GUIDELINE_COLOR, fontweight="bold")
    ax.text(4.05, 104, "Zero-Shot", ha="center", fontsize=9,
            color=ZEROSHOT_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, alpha=0.35, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_title("Figure 1. CLiTR-Bench CMS125 — Model Performance Across All Conditions\n"
                 "(n = 500 patients; error bars on F1 = 95% bootstrap CI, B = 10,000)", pad=12)
    ax.legend(loc="upper right", ncol=5, framealpha=0.9, fontsize=9,
              columnspacing=0.8, handlelength=1.2)

    out = os.path.join(FIGURES_DIR, "figure1_performance_bars.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Stacked bar: Zero-shot error taxonomy comparison
# ═══════════════════════════════════════════════════════════════════════════════
def figure2_error_taxonomy():
    models        = ["Qwen 3 80B\nZero-Shot\n(n=282 non-auditable)", "GPT-4o\nZero-Shot\n(n=237 non-auditable)"]
    wrong_conc    = [92.9, 51.1]
    outside_win   = [2.5,  36.3]
    no_evidence   = [4.6,  12.7]

    x    = np.arange(len(models))
    w    = 0.48
    c1, c2, c3 = "#E84855", "#F18F01", "#C0C0C0"

    fig, ax = plt.subplots(figsize=(8, 5.5))

    b1 = ax.bar(x, wrong_conc,  width=w, label="WRONG_CONCLUSION",  color=c1, alpha=0.9)
    b2 = ax.bar(x, outside_win, width=w, label="OUTSIDE_WINDOW",    color=c2, alpha=0.9,
                bottom=wrong_conc)
    b3 = ax.bar(x, no_evidence, width=w, label="NO_EVIDENCE_CITED", color=c3, alpha=0.9,
                bottom=[a + b for a, b in zip(wrong_conc, outside_win)])

    # Segment labels
    for i, (wc, ow, ne) in enumerate(zip(wrong_conc, outside_win, no_evidence)):
        ax.text(x[i], wc / 2,               f"{wc:.1f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x[i], wc + ow / 2,          f"{ow:.1f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white" if i == 1 else "#333")
        ax.text(x[i], wc + ow + ne / 2,     f"{ne:.1f}%", ha="center", va="center",
                fontsize=9,  color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("% of Non-Auditable Cases", fontsize=10)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_title("Figure 2. Zero-Shot Hallucination Taxonomy — Error Type Distribution\n"
                 "by Model (% of Non-Auditable Patients per Condition)", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax.annotate("Temporal boundary\nblindness — never\nchecks the window",
                xy=(0, 50), xytext=(-0.3, 70),
                fontsize=8, color="#333", ha="right",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    ax.annotate("Miscalibrated — attempts\nthe window check\nbut uses wrong cutoff",
                xy=(1, 51.1 + 18), xytext=(1.28, 82),
                fontsize=8, color="#333", ha="left",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

    out = os.path.join(FIGURES_DIR, "figure2_error_taxonomy.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CONSORT-style cohort derivation flowchart
# ═══════════════════════════════════════════════════════════════════════════════
def figure3_consort_flowchart():
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(ax, x, y, w, h, text, color="#DBEAFE", textcolor="#1e3a5f", fontsize=9.5):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor="#2E86AB",
                                        linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor,
                fontweight="normal", wrap=True,
                multialignment="center")

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2E86AB",
                                   lw=1.5, mutation_scale=14))

    def excl_box(ax, x, y, text):
        ax.text(x, y, text, ha="left", va="center", fontsize=8.5,
                color="#7f1d1d",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2",
                          edgecolor="#EF4444", linewidth=1.1))

    # Boxes
    box(ax, 5, 11,   8.5, 0.8, "Synthea™ simulation — 25,000 California patients", "#EFF6FF")
    arrow(ax, 5, 10.6, 5, 10.0)
    box(ax, 5, 9.7,  8.5, 0.7,
        "Initial population screen: Female, aged 52–74 as of 2025-12-31\n→ 3,002 patients")
    arrow(ax, 5, 9.35, 5, 8.75)
    excl_box(ax, 5.6, 9.1, "Excluded: 21,998 patients (wrong sex, age, or both)")

    box(ax, 5, 8.45, 8.5, 0.7,
        "Denominator criteria: ≥1 qualifying 2025 encounter\n→ 2,987 patients")
    arrow(ax, 5, 8.10, 5, 7.50)
    excl_box(ax, 5.6, 7.75, "Excluded: 15 patients (no qualifying encounter)")

    box(ax, 5, 7.20, 8.5, 0.7,
        "Exclusion criteria: bilateral mastectomy history\n→ 2,987 patients (0 excluded)")
    arrow(ax, 5, 6.85, 5, 6.25)
    excl_box(ax, 5.6, 6.55, "Excluded: 0 patients (no mastectomy coded in Synthea)")

    box(ax, 5, 5.95, 8.5, 0.85,
        "Gold truth evaluation\nCompliant (mammogram in 27-month window): 147 (4.9%)\nNon-compliant: 2,840 (95.1%)", "#F0FDF4")

    # Split arrows
    arrow(ax, 3.5, 5.53, 2.5, 4.65)
    arrow(ax, 6.5, 5.53, 7.5, 4.65)

    box(ax, 2.5, 4.3,  3.8, 0.65, "All 147 compliant patients\nincluded", "#F0FDF4")
    box(ax, 7.5, 4.3,  3.8, 0.65, "352 non-compliant patients\nrandom sample (seed=99)", "#FFF7ED")

    # Merge arrows
    arrow(ax, 2.5, 3.97, 4.0, 3.2)
    arrow(ax, 7.5, 3.97, 6.0, 3.2)

    box(ax, 5, 2.85, 8.5, 0.65,
        "Publication cohort — n = 499 patients (147 compliant, 352 non-compliant)\nCompliance rate: 29.4% (vs. 4.9% natural rate — disclosed)", "#EFF6FF")

    arrow(ax, 5, 2.52, 5, 1.85)

    box(ax, 5, 1.55, 8.5, 0.55,
        "LLM inference: 6 conditions × 500 calls = 3,000 total API calls\n"
        "GPT-4o GS · Llama 3.3 70B GS · Qwen 3 80B GS · GPT-4o ZSB · Llama ZSB · Qwen ZSB",
        "#F5F3FF")

    ax.set_title("Figure 3. Cohort Derivation Flowchart (CONSORT-style)\n"
                 "CMS125 HEDIS 2025 BCS-E — Synthea™ Synthetic Patient Population",
                 fontsize=11, fontweight="bold", pad=10)

    out = os.path.join(FIGURES_DIR, "figure3_consort_flowchart.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — AMR comparison: Guideline vs Zero-Shot (lollipop chart)
# ═══════════════════════════════════════════════════════════════════════════════
def figure4_amr_hallucinations():
    models  = ["GPT-4o", "Llama 3.3 70B", "Qwen 3 80B"]
    amr_gs  = [98.0, 92.4, 82.6]
    amr_zs  = [35.0, 69.8, 43.6]
    hall_gs = [10,   38,   89]
    hall_zs = [325,  196,  282]

    x = np.arange(len(models))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: AMR grouped bars
    w = 0.32
    bars1 = ax1.bar(x - w/2, amr_gs, width=w, label="Guideline-Supplied",
                    color=GUIDELINE_COLOR, alpha=0.88)
    bars2 = ax1.bar(x + w/2, amr_zs, width=w, label="Zero-Shot",
                    color=ZEROSHOT_COLOR, alpha=0.88)
    for bar, v in [(b, va) for bars, vals in [(bars1, amr_gs), (bars2, amr_zs)]
                   for b, va in zip(bars, vals)]:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("Auditability Match Rate (%)")
    ax1.set_ylim(0, 110)
    ax1.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax1.set_axisbelow(True)
    ax1.set_title("Auditability Match Rate\nGuideline-Supplied vs Zero-Shot", pad=8)
    ax1.legend(framealpha=0.9, fontsize=9)
    ax1.axhline(100, color="#aaa", linestyle=":", linewidth=0.8)

    # Right: Hallucinations grouped bars
    bars3 = ax2.bar(x - w/2, hall_gs, width=w, label="Guideline-Supplied",
                    color=GUIDELINE_COLOR, alpha=0.88)
    bars4 = ax2.bar(x + w/2, hall_zs, width=w, label="Zero-Shot",
                    color=ZEROSHOT_COLOR, alpha=0.88)
    for bar, v in [(b, va) for bars, vals in [(bars3, hall_gs), (bars4, hall_zs)]
                   for b, va in zip(bars, vals)]:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(v), ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("Hallucinated Evidence Citations (n)")
    ax2.set_ylim(0, 370)
    ax2.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.set_title("Hallucinated Evidence Citations\nGuideline-Supplied vs Zero-Shot", pad=8)
    ax2.legend(framealpha=0.9, fontsize=9)

    fig.suptitle("Figure 4. Auditability Match Rate and Hallucination Counts by Prompt Strategy",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "figure4_amr_hallucinations.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating CLiTR-Bench publication figures...")
    figure1_grouped_bars()
    figure2_error_taxonomy()
    figure3_consort_flowchart()
    figure4_amr_hallucinations()
    print(f"\nAll figures saved to: {FIGURES_DIR}")
