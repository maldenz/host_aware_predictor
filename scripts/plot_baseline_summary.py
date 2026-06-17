from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("runs/baseline_summary_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

records = [
    # head, condition, regime, n, mse, rmse, mae, r2, pearson, spearman, loss
    ("concat", "all_conditions", "all_conditions_global", 16962, 0.6044509769, 0.7774644538, 0.5872545730, 0.2280310031, 0.4780610249, 0.4774005163, 0.7880585763),
    ("concat", "HepG2", "single_cell", 5654, 0.4476991699, 0.6691032580, 0.5246562974, 0.1780674925, 0.4236456066, 0.3935865582, 0.8246064601),
    ("concat", "K562", "single_cell", 5654, 0.4783135747, 0.6916021795, 0.5320499580, 0.2123232417, 0.4634351683, 0.4085166592, 0.8046865217),
    ("concat", "WTC11", "single_cell", 5654, 0.8041207067, 0.8967277774, 0.6531347014, 0.2739827636, 0.5238093668, 0.5154070802, 0.7339115868),

    ("film", "all_conditions", "all_conditions_global", 16962, 0.5518978601, 0.7428982838, 0.5579486773, 0.2951487321, 0.5435401538, 0.5187504608, 0.7192256949),
    ("film", "HepG2", "single_cell", 5654, 0.4486197292, 0.6697908100, 0.5264018263, 0.1763774345, 0.4210961309, 0.3835789320, 0.8258244214),
    ("film", "K562", "single_cell", 5654, 0.4581749611, 0.6768862246, 0.5207091330, 0.2454870879, 0.4955500069, 0.4242465391, 0.7728739122),
    ("film", "WTC11", "single_cell", 5654, 0.8017509946, 0.8954054917, 0.6592024666, 0.2761223079, 0.5277876520, 0.5128047552, 0.7305026158),

    ("query", "all_conditions", "all_conditions_global", 16962, 0.5587397487, 0.7474889622, 0.5572210972, 0.2864106772, 0.5356788457, 0.5176254428, 0.7286150971),
    ("query", "HepG2", "single_cell", 5654, 0.4364043455, 0.6606090716, 0.5178590912, 0.1988037010, 0.4483579742, 0.4084362874, 0.8060685552),
    ("query", "K562", "single_cell", 5654, 0.4538484232, 0.6736827319, 0.5177776173, 0.2526119398, 0.5026651665, 0.4337995727, 0.7652611149),
    ("query", "WTC11", "single_cell", 5654, 0.7773445589, 0.8816714575, 0.6418215307, 0.2981581699, 0.5465259615, 0.5271071818, 0.7081519935),
]

df = pd.DataFrame(
    records,
    columns=[
        "head",
        "condition",
        "regime",
        "n",
        "mse",
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
        "loss",
    ],
)

df.to_csv(OUT_DIR / "baseline_test_metrics.tsv", sep="\t", index=False)


# Plot 1: all runs ranked by test Pearson.
ranked = df.sort_values("pearson", ascending=True).copy()
ranked["label"] = ranked["head"] + " / " + ranked["condition"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ranked["pearson"], ranked["label"])
ax.set_xlabel("Test Pearson")
ax.set_ylabel("Run")
ax.set_title("Baseline runs ranked by test Pearson")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "all_runs_ranked_test_pearson.png", dpi=200)
plt.close(fig)


# Plot 2: single-cell Pearson heatmap.
single = df[df["regime"] == "single_cell"].copy()
cell_order = ["HepG2", "K562", "WTC11"]
head_order = ["concat", "film", "query"]

pearson_matrix = (
    single.pivot(index="condition", columns="head", values="pearson")
    .loc[cell_order, head_order]
)

fig, ax = plt.subplots(figsize=(5, 3.5))
image = ax.imshow(pearson_matrix.values)
ax.set_xticks(range(len(head_order)))
ax.set_xticklabels(head_order)
ax.set_yticks(range(len(cell_order)))
ax.set_yticklabels(cell_order)
ax.set_title("Single-cell test Pearson")
fig.colorbar(image, ax=ax, label="Pearson")

for row_idx, condition in enumerate(cell_order):
    for col_idx, head in enumerate(head_order):
        value = pearson_matrix.loc[condition, head]
        ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center")

fig.tight_layout()
fig.savefig(OUT_DIR / "single_cell_test_pearson_heatmap.png", dpi=200)
plt.close(fig)


# Plot 3: single-cell RMSE heatmap.
rmse_matrix = (
    single.pivot(index="condition", columns="head", values="rmse")
    .loc[cell_order, head_order]
)

fig, ax = plt.subplots(figsize=(5, 3.5))
image = ax.imshow(rmse_matrix.values)
ax.set_xticks(range(len(head_order)))
ax.set_xticklabels(head_order)
ax.set_yticks(range(len(cell_order)))
ax.set_yticklabels(cell_order)
ax.set_title("Single-cell test RMSE")
fig.colorbar(image, ax=ax, label="RMSE")

for row_idx, condition in enumerate(cell_order):
    for col_idx, head in enumerate(head_order):
        value = rmse_matrix.loc[condition, head]
        ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center")

fig.tight_layout()
fig.savefig(OUT_DIR / "single_cell_test_rmse_heatmap.png", dpi=200)
plt.close(fig)


# Plot 4: all-condition global head comparison.
all_conditions = df[df["regime"] == "all_conditions_global"].copy()
all_conditions = all_conditions.set_index("head").loc[head_order].reset_index()

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(all_conditions["head"], all_conditions["pearson"])
ax.set_xlabel("Head")
ax.set_ylabel("Test Pearson")
ax.set_title("All-condition global test Pearson")
ax.grid(axis="y", alpha=0.3)

for _, row in all_conditions.iterrows():
    ax.text(row["head"], row["pearson"], f"{row['pearson']:.3f}", ha="center", va="bottom")

fig.tight_layout()
fig.savefig(OUT_DIR / "all_conditions_global_test_pearson.png", dpi=200)
plt.close(fig)

print(f"Wrote plots and table to: {OUT_DIR}")