import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = "final_hallucination_evaluation_126_questions.csv"

# ---------------------------------------------------------------------------
# Chart style settings (consistent, professional look for all figures)
# ---------------------------------------------------------------------------
LABEL_ORDER = ["supported", "partially_supported", "hallucinated", "contradictory"]
PRETTY = {
    "supported": "Supported",
    "partially_supported": "Partially supported",
    "hallucinated": "Hallucinated",
    "contradictory": "Contradictory",
}
# semantic palette: green -> amber -> orange -> red
LABEL_COLORS = {
    "supported": "#4c8a5a",
    "partially_supported": "#d9a441",
    "hallucinated": "#d06a3f",
    "contradictory": "#a63d3d",
}
NEUTRAL = ["#4a6fa5", "#8a9bb8", "#c3cbd9"]

plt.rcParams.update({
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def style_axes(ax, ylabel, ymax=None):
    """Remove box lines, add subtle gridlines, set labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)


def add_bar_labels(ax, bars, fmt="{:.1f}%"):
    """Print the value above each bar."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10.5,
                    fontweight="bold", color="#333333",
                    xytext=(0, 2), textcoords="offset points")


def grouped_bar_chart(data, title, xlabel, ylabel, fname, fmt, group_labels):
    """Grouped bar chart with one colored bar per hallucination label."""
    x = np.arange(len(data.index))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, lab in enumerate(LABEL_ORDER):
        bars = ax.bar(x + (i - 1.5) * width, data[lab].values, width,
                      label=PRETTY[lab], color=LABEL_COLORS[lab],
                      edgecolor="#333333", linewidth=0.5)
        add_bar_labels(ax, bars, fmt=fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    style_axes(ax, ylabel, ymax=data.values.max() * 1.22)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(ncol=2, loc="upper right", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)

# Load final hallucination evaluation file
df = pd.read_csv(INPUT_FILE)

# Make sure labels are consistent
df["hallucination_label"] = df["hallucination_label"].astype(str).str.strip().str.lower()
df["question_type"] = df["question_type"].astype(str).str.strip().str.lower()
df["source"] = df["source"].astype(str).str.strip()
df = df[df["hallucination_label"].isin(LABEL_ORDER)]
n_total = len(df)

# 1. Overall summary

overall_counts = df["hallucination_label"].value_counts().reindex(LABEL_ORDER).reset_index()
overall_counts.columns = ["hallucination_label", "count"]
overall_counts["percentage"] = (overall_counts["count"] / len(df) * 100).round(2)

overall_counts.to_csv("results_overall_hallucination_counts.csv", index=False)

print("\nOverall hallucination counts:")
print(overall_counts)

# Bar chart: Overall hallucination labels
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([PRETTY[l] for l in overall_counts["hallucination_label"]],
              overall_counts["count"],
              color=[LABEL_COLORS[l] for l in overall_counts["hallucination_label"]],
              width=0.6, edgecolor="#333333", linewidth=0.6)
for b, c, p in zip(bars, overall_counts["count"], overall_counts["percentage"]):
    ax.annotate(f"{c}\n({p:.1f}%)", (b.get_x() + b.get_width() / 2, c),
                ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color="#333333",
                xytext=(0, 2), textcoords="offset points")
style_axes(ax, "Number of Responses", ymax=overall_counts["count"].max() * 1.25)
ax.set_xlabel("Hallucination Label")
ax.set_title(f"Overall Hallucination Evaluation Results (n = {n_total})")
plt.tight_layout()
plt.savefig("chart_overall_hallucination_counts.png", bbox_inches="tight")
plt.close(fig)

# Pie chart: Overall distribution
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(
    overall_counts["count"],
    labels=[PRETTY[l] for l in overall_counts["hallucination_label"]],
    colors=[LABEL_COLORS[l] for l in overall_counts["hallucination_label"]],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 11.5},
)
ax.set_title(f"Overall Hallucination Label Distribution (n = {n_total})", fontweight="bold")
plt.tight_layout()
plt.savefig("chart_overall_hallucination_pie.png", bbox_inches="tight")
plt.close(fig)

# 2. Hallucination by question type

type_counts = pd.crosstab(
    df["question_type"],
    df["hallucination_label"]
)[LABEL_ORDER].reindex(["factual", "open_ended"])

type_percent = (type_counts.div(type_counts.sum(axis=1), axis=0) * 100)
type_group_labels = [f"Factual (n={type_counts.iloc[0].sum()})",
                     f"Open-ended (n={type_counts.iloc[1].sum()})"]

type_counts.to_csv("results_hallucination_by_question_type_counts.csv")
type_percent.round(2).to_csv("results_hallucination_by_question_type_percent.csv")

print("\nHallucination by question type counts:")
print(type_counts)

print("\nHallucination by question type percentage:")
print(type_percent.round(2))

# Bar chart: counts by question type
grouped_bar_chart(type_counts, "Hallucination Labels by Question Type",
                  "Question Type", "Number of Responses",
                  "chart_hallucination_by_question_type_counts.png",
                  "{:.0f}", type_group_labels)

# Bar chart: percentage by question type
grouped_bar_chart(type_percent, "Hallucination Label Percentages by Question Type",
                  "Question Type", "Percentage of Responses (%)",
                  "chart_hallucination_by_question_type_percent.png",
                  "{:.1f}%", type_group_labels)


# 3. Hallucination by source

source_counts = pd.crosstab(
    df["source"],
    df["hallucination_label"]
)[LABEL_ORDER].reindex(["CRA", "OSC", "Reddit"])

source_percent = (source_counts.div(source_counts.sum(axis=1), axis=0) * 100)
source_group_labels = [f"{s} (n={source_counts.loc[s].sum()})" for s in source_counts.index]

source_counts.to_csv("results_hallucination_by_source_counts.csv")
source_percent.round(2).to_csv("results_hallucination_by_source_percent.csv")

print("\nHallucination by source counts:")
print(source_counts)

print("\nHallucination by source percentage:")
print(source_percent.round(2))

# Bar chart: counts by source
grouped_bar_chart(source_counts, "Hallucination Labels by Source",
                  "Source", "Number of Responses",
                  "chart_hallucination_by_source_counts.png",
                  "{:.0f}", source_group_labels)

# Bar chart: percentage by source
grouped_bar_chart(source_percent, "Hallucination Label Percentages by Source",
                  "Source", "Percentage of Responses (%)",
                  "chart_hallucination_by_source_percent.png",
                  "{:.1f}%", source_group_labels)

# 4. Dataset composition charts
question_type_distribution = df["question_type"].value_counts().reset_index()
question_type_distribution.columns = ["question_type", "count"]
question_type_distribution.to_csv("results_question_type_distribution.csv", index=False)

qt_names = {"open_ended": "Open-ended", "factual": "Factual"}
question_type_distribution["percentage"] = (
    question_type_distribution["count"]
    / question_type_distribution["count"].sum() * 100
).round(1)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    [qt_names.get(q, q) for q in question_type_distribution["question_type"]],
    question_type_distribution["percentage"],
    color=NEUTRAL[: len(question_type_distribution)],
    width=0.5, edgecolor="#333333", linewidth=0.6,
)
add_bar_labels(ax, bars)
style_axes(ax, "Percentage of Questions (%)",
           ymax=question_type_distribution["percentage"].max() * 1.18)
ax.set_xlabel("Question Type")
ax.set_title(f"Distribution of Factual and Open-Ended Questions (n = {n_total})")
plt.tight_layout()
plt.savefig("chart_question_type_distribution.png", bbox_inches="tight")
plt.close(fig)

source_distribution = df["source"].value_counts().reset_index()
source_distribution.columns = ["source", "count"]
source_distribution["percentage"] = (
        source_distribution["count"] / source_distribution["count"].sum() * 100
).round(1)

source_distribution.to_csv("results_source_distribution.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    source_distribution["source"],
    source_distribution["percentage"],
    color=NEUTRAL[: len(source_distribution)],
    width=0.55, edgecolor="#333333", linewidth=0.6,
)
add_bar_labels(ax, bars)
style_axes(ax, "Percentage of Questions (%)",
           ymax=source_distribution["percentage"].max() * 1.18)
ax.set_xlabel("Source")
ax.set_title(f"Distribution of Questions by Source (n = {n_total})")
plt.tight_layout()
plt.savefig("chart_source_distribution.png", bbox_inches="tight")
plt.close(fig)

# 5. Model comparison chart (uses step 3 output if available)
if os.path.exists("results_model_comparison.csv"):
    comp = pd.read_csv("results_model_comparison.csv")
    model_pct = comp.set_index("model")[[f"{l}_pct" for l in LABEL_ORDER]]
    model_pct.columns = LABEL_ORDER
    model_group_labels = [f"{m}\n(n={int(n)})" for m, n in zip(comp["model"], comp["n"])]
    grouped_bar_chart(model_pct, "Hallucination Label Percentages by Model",
                      "Model", "Percentage of Responses (%)",
                      "chart_model_comparison.png",
                      "{:.1f}%", model_group_labels)
    print("\nModel comparison chart saved -> chart_model_comparison.png")
else:
    print("\nresults_model_comparison.csv not found - run step 3 first, then rerun this script.")

print("\nDone. Charts and result tables have been saved.")

