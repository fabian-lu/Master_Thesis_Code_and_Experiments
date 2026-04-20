import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

# ── Configuration ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ── Raw logits (simulated for a vocabulary of 10 representative tokens) ───
token_labels = [
    '"the"', '"a"', '"an"', '"this"', '"one"',
    '"some"', '"that"', '"my"', '"its"', '"our"'
]
# Logits designed so that "the" is strongly preferred, with a clear ranking
logits = np.array([5.2, 3.8, 3.1, 2.5, 1.9, 1.4, 0.8, 0.3, -0.2, -0.8])

# ── Softmax with temperature ─────────────────────────────────────────────
def softmax(logits, temperature):
    scaled = logits / temperature
    scaled -= scaled.max()  # numerical stability
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum()

temperatures = [0.3, 0.7, 1.0, 1.5, 3.0]
temp_labels  = [r"$\tau = 0.3$", r"$\tau = 0.7$", r"$\tau = 1.0$", r"$\tau = 1.5$", r"$\tau = 3.0$"]

# Color palette: cool→warm  (blue = cold/sharp, red = hot/flat)
colors = ["#154360", "#2980b9", "#7f8c8d", "#e67e22", "#c0392b"]

# ── Compute distributions ────────────────────────────────────────────────
distributions = [softmax(logits, t) for t in temperatures]

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.0, 5.2))

x = np.arange(len(token_labels))
n = len(temperatures)
bar_width = 0.145
offsets = np.linspace(-(n-1)/2 * bar_width, (n-1)/2 * bar_width, n)

for i, (dist, label, color) in enumerate(zip(distributions, temp_labels, colors)):
    ax.bar(x + offsets[i], dist, width=bar_width, label=label,
           color=color, edgecolor="white", linewidth=0.5, alpha=0.90,
           zorder=3)

# ── Axes & labels ─────────────────────────────────────────────────────────
ax.set_xlabel("Token (ranked by logit magnitude)", fontweight="medium", labelpad=8)
ax.set_ylabel("Probability  $p_i = \\mathrm{softmax}(z_i / \\tau)$",
              fontweight="medium", labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(token_labels, rotation=0, ha="center", fontsize=10,
                   fontstyle="italic")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# Grid
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.55, color="#aaaaaa")
ax.xaxis.grid(False)

# Spine cleanup
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#555555")
ax.spines["bottom"].set_color("#555555")

# ── Legend ────────────────────────────────────────────────────────────────
legend = ax.legend(
    title="Temperature ($\\tau$)", title_fontsize=11,
    loc="upper right", frameon=True, framealpha=0.95,
    edgecolor="#bbbbbb", fancybox=False,
    ncol=1, handlelength=1.4, handletextpad=0.5,
    borderpad=0.6, labelspacing=0.4,
)
legend.get_frame().set_linewidth(0.6)

# ── Annotations ──────────────────────────────────────────────────────────
# "Low T: concentrated" — pointing at T=0.3 bar for "the"
peak_val = distributions[0][0]
ax.annotate(
    "Low $\\tau$: probability\nconcentrated on top token",
    xy=(x[0] + offsets[0], peak_val),
    xytext=(2.2, 0.82),
    fontsize=9.5, fontstyle="italic", color=colors[0],
    arrowprops=dict(arrowstyle="-|>", color=colors[0], lw=1.3,
                    connectionstyle="arc3,rad=-0.15"),
    ha="left", va="top", zorder=5,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colors[0],
              lw=0.6, alpha=0.85),
)

# "High T: flattened" — pointing at T=3.0 bar for "the"
flat_val = distributions[-1][0]
ax.annotate(
    "High $\\tau$: probability\nspread across all tokens",
    xy=(x[0] + offsets[-1], flat_val),
    xytext=(2.2, 0.38),
    fontsize=9.5, fontstyle="italic", color=colors[-1],
    arrowprops=dict(arrowstyle="-|>", color=colors[-1], lw=1.3,
                    connectionstyle="arc3,rad=0.15"),
    ha="left", va="bottom", zorder=5,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colors[-1],
              lw=0.6, alpha=0.85),
)

# ── Final adjustments ────────────────────────────────────────────────────
ax.set_ylim(0, 1.05)
ax.set_xlim(-0.6, len(token_labels) - 0.4)
plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────
out = "/home/fabian/Desktop/Master_thesis/Thesis_latex/figures/figures_ch2"
fig.savefig(f"{out}/fig_temperature_effect.png", transparent=False,
            facecolor="white")
fig.savefig(f"{out}/fig_temperature_effect.pdf")
print("Saved PNG (600 dpi) and PDF.")
plt.close()
