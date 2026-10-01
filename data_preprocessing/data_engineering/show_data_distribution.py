import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

LABEL_COL          = "建議"
BMI_COL            = "BMI"
POSTOP_DAYS_COL    = "術後天數"
WOUND_SIZE_COL     = "傷口大小"
CRACK_COL          = "裂開"
EXUDATE_COL        = "滲液"
SWELL_SEVERITY_COL = "紅腫程度"
SWELL_EXTENT_COL   = "紅腫範圍"

NORMAL_LABELS   = ["正常，冰敷即可", "低風險，加強冰敷"]
ABNORMAL_LABELS = ["中度風險，觀察3天", "高風險，立即就醫"]

CRACK_ORDER          = ["無", "有"]
EXUDATE_ORDER        = ["無", "有-淡黃清澈", "有-深黃黏稠", "有-暗紅褐色", "有-鮮血"]
SWELL_SEVERITY_ORDER = ["無", "淡紅，局部", "深紅，大範圍"]
SWELL_EXTENT_ORDER   = ["無", "紅腫<10公分", "紅腫>10公分"]

CRACK_VALUE_NAMES = {"無": "None", "有": "Present"}
EXUDATE_VALUE_NAMES = {
    "無": "None",
    "有-淡黃清澈": "Pale Yellow, Clear",
    "有-深黃黏稠": "Dark Yellow, Viscous",
    "有-暗紅褐色": "Dark Red-Brown",
    "有-鮮血": "Fresh Blood",
}
SWELL_SEVERITY_VALUE_NAMES = {
    "無": "None",
    "淡紅，局部": "Mild, Localized",
    "深紅，大範圍": "Severe, Extensive",
}
SWELL_EXTENT_VALUE_NAMES = {
    "無": "None",
    "紅腫<10公分": "<10cm",
    "紅腫>10公分": ">10cm",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot distributions of key wound-metadata variables.")
    parser.add_argument("--metadata", type=str, required=True, help="path to master_metadata.csv")
    return parser.parse_args()

def _annotate_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                ha="center", va="bottom", fontsize=12, fontweight="bold")

def _annotate_stack_totals(ax, top_bars, totals):
    for bar, total in zip(top_bars, totals):
        top = bar.get_y() + bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, top, str(total),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

def _ordered_labels(counts, preferred_order):
    return [l for l in preferred_order if l in counts.index] + \
           [l for l in counts.index if l not in preferred_order]

def _plot_risk_group_bar(ax, df):
    grouped = df[LABEL_COL].apply(lambda v: "Normal" if v in NORMAL_LABELS else "Abnormal")
    counts = grouped.value_counts()
    labels = ["Normal", "Abnormal"]
    values = [int(counts.get(l, 0)) for l in labels]
    bars = ax.bar(labels, values, color=["#4CAF50", "#E53935"])
    _annotate_bars(ax, bars, values)
    ax.set_title(f"Risk Group Distribution (n={len(df)})", fontsize=18)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)

def _plot_numeric_hist(ax, df, col, display_name, bins=20):
    normal_data   = df.loc[df[LABEL_COL].isin(NORMAL_LABELS), col].dropna()
    abnormal_data = df.loc[df[LABEL_COL].isin(ABNORMAL_LABELS), col].dropna()
    ax.hist([normal_data, abnormal_data], bins=bins, stacked=True,
            color=["#4CAF50", "#E53935"], label=["Normal", "Abnormal"], edgecolor="white")
    ax.set_title(f"{display_name} Distribution", fontsize=18)
    ax.set_xlabel(display_name, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.legend(fontsize=18, loc="best")

def _plot_categorical_bar(ax, df, col, order, value_names, display_name):
    normal_counts   = df.loc[df[LABEL_COL].isin(NORMAL_LABELS), col].value_counts()
    abnormal_counts = df.loc[df[LABEL_COL].isin(ABNORMAL_LABELS), col].value_counts()
    total_counts    = df[col].value_counts()

    labels          = _ordered_labels(total_counts, order)
    display_labels  = [value_names.get(l, l) for l in labels]
    normal_values   = [int(normal_counts.get(l, 0)) for l in labels]
    abnormal_values = [int(abnormal_counts.get(l, 0)) for l in labels]
    totals          = [n + a for n, a in zip(normal_values, abnormal_values)]

    ax.bar(display_labels, normal_values, color="#4CAF50", label="Normal")
    abnormal_bars = ax.bar(display_labels, abnormal_values, bottom=normal_values, color="#E53935", label="Abnormal")

    _annotate_stack_totals(ax, abnormal_bars, totals)
    ax.set_title(f"{display_name} Distribution (n={len(df)})")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()

def _plot_overview(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_risk_group_bar(axes[0, 0], df)
    _plot_numeric_hist(axes[0, 1], df, BMI_COL, "BMI")
    _plot_numeric_hist(axes[1, 0], df, POSTOP_DAYS_COL, "Post-Op Days")
    _plot_numeric_hist(axes[1, 1], df, WOUND_SIZE_COL, "Wound Size")
    fig.suptitle(f"Overview Distribution (n={len(df)})", fontsize=20)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "overview_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Overview distribution plot saved to {save_path}")

def _plot_single_categorical(df, output_dir, col, order, value_names, display_name, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_categorical_bar(ax, df, col, order, value_names, display_name)
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{display_name} distribution plot saved to {save_path}")

def generate_distribution_plots(csv_path):
    df = pd.read_csv(csv_path)

    required = [LABEL_COL, BMI_COL, POSTOP_DAYS_COL, WOUND_SIZE_COL,
                CRACK_COL, EXUDATE_COL, SWELL_SEVERITY_COL, SWELL_EXTENT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in {csv_path}")

    output_dir = os.path.dirname(os.path.abspath(csv_path))

    _plot_overview(df, output_dir)
    _plot_single_categorical(df, output_dir, CRACK_COL, CRACK_ORDER, CRACK_VALUE_NAMES,
                              "Wound Dehiscence", "crack_distribution.png")
    _plot_single_categorical(df, output_dir, EXUDATE_COL, EXUDATE_ORDER, EXUDATE_VALUE_NAMES,
                              "Exudate", "exudate_distribution.png")
    _plot_single_categorical(df, output_dir, SWELL_SEVERITY_COL, SWELL_SEVERITY_ORDER, SWELL_SEVERITY_VALUE_NAMES,
                              "Redness Severity", "swelling_severity_distribution.png")
    _plot_single_categorical(df, output_dir, SWELL_EXTENT_COL, SWELL_EXTENT_ORDER, SWELL_EXTENT_VALUE_NAMES,
                              "Redness Extent", "swelling_extent_distribution.png")

if __name__ == "__main__":
    args = parse_args()
    generate_distribution_plots(args.metadata)
