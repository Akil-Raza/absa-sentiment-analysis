"""
eda.py
--------
Exploratory data analysis charts: class balance, sentence length
distribution, and most frequent aspect terms.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Polarity class balance (train vs test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, df, title in zip(axes, [train_df, test_df], ["Train", "Test"]):
        order = ["negative", "neutral", "positive"]
        counts = df["polarity_label"].value_counts().reindex(order)
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                    palette=["#d62728", "#7f7f7f", "#2ca02c"], legend=False, ax=ax)
        ax.set_title(f"{title} set polarity distribution")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/class_balance.png", dpi=150)
    plt.close()

    # 2. Sentence length distribution
    plt.figure(figsize=(7, 4))
    train_df["sentence"].str.split().apply(len).hist(bins=30, color="#4c72b0")
    plt.title("Sentence length distribution (train)")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/sentence_length.png", dpi=150)
    plt.close()

    # 3. Most frequent raw aspect terms
    plt.figure(figsize=(7, 6))
    top_aspects = train_df["aspect"].value_counts().head(20)
    sns.barplot(x=top_aspects.values, y=top_aspects.index, hue=top_aspects.index,
                palette="mako", legend=False)
    plt.title("Top 20 most frequent aspect terms (train)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/top_aspects.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    run_eda(train_df, test_df)
    print("EDA figures saved to figures/")
