"""
business_insights.py
----------------------
Turns per-review, per-aspect sentiment predictions into the kind of
rollup a business stakeholder actually wants:

  - Which aspect categories get the most mentions?
  - Which categories have the worst sentiment mix?
  - What's the overall "net sentiment score" per category?

Produces charts (saved to figures/) and a summary table
(saved to reports/business_insights_summary.csv).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aspect_categories import add_category_column

sns.set_theme(style="whitegrid")


def net_sentiment_score(group: pd.Series) -> float:
    """(% positive - % negative), a common single-number sentiment metric
    used in review analytics dashboards (range: -100 to +100)."""
    counts = group.value_counts(normalize=True) * 100
    pos = counts.get("positive", 0.0)
    neg = counts.get("negative", 0.0)
    return round(pos - neg, 1)


def build_summary(df: pd.DataFrame, sentiment_col: str = "polarity_label") -> pd.DataFrame:
    df = add_category_column(df)
    grouped = df.groupby("aspect_category")
    summary = grouped.agg(
        mentions=(sentiment_col, "count"),
        pct_positive=(sentiment_col, lambda s: round((s == "positive").mean() * 100, 1)),
        pct_neutral=(sentiment_col, lambda s: round((s == "neutral").mean() * 100, 1)),
        pct_negative=(sentiment_col, lambda s: round((s == "negative").mean() * 100, 1)),
    ).reset_index()
    summary["net_sentiment_score"] = grouped[sentiment_col].apply(net_sentiment_score).values
    summary = summary.sort_values("mentions", ascending=False)
    return summary


def plot_mentions_and_sentiment(summary: pd.DataFrame, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)

    # Chart 1: mention volume per category
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="mentions", y="aspect_category", hue="aspect_category",
                palette="viridis", legend=False)
    plt.title("Review Mentions by Aspect Category")
    plt.xlabel("Number of mentions")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mentions_by_category.png", dpi=150)
    plt.close()

    # Chart 2: net sentiment score per category
    plt.figure(figsize=(8, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in summary["net_sentiment_score"]]
    order = summary.sort_values("net_sentiment_score")
    plt.barh(order["aspect_category"], order["net_sentiment_score"],
             color=["#d62728" if v < 0 else "#2ca02c" for v in order["net_sentiment_score"]])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Net Sentiment Score by Aspect Category (%pos - %neg)")
    plt.xlabel("Net sentiment score")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/net_sentiment_by_category.png", dpi=150)
    plt.close()

    # Chart 3: stacked sentiment mix
    plot_df = summary.set_index("aspect_category")[["pct_positive", "pct_neutral", "pct_negative"]]
    plot_df.plot(
        kind="barh", stacked=True, figsize=(8, 5),
        color=["#2ca02c", "#7f7f7f", "#d62728"]
    )
    plt.title("Sentiment Mix by Aspect Category")
    plt.xlabel("Percentage of mentions")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/sentiment_mix_by_category.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("data/processed/test.csv")
    summary = build_summary(df)
    os.makedirs("reports", exist_ok=True)
    summary.to_csv("reports/business_insights_summary.csv", index=False)
    plot_mentions_and_sentiment(summary)
    print(summary)
