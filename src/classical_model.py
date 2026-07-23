"""
classical_model.py
---------------------
Baseline aspect-based sentiment classifier using TF-IDF features and
two classical models (Logistic Regression, Linear SVM). This is the
"Data Analyst" layer of the project: interpretable features, clear
metrics, quick to train on a laptop, no GPU required.

Usage:
    python src/classical_model.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocessing import basic_clean, make_aspect_aware_input


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_text = [
        basic_clean(make_aspect_aware_input(s, a))
        for s, a in zip(train_df["sentence"], train_df["aspect"])
    ]
    test_text = [
        basic_clean(make_aspect_aware_input(s, a))
        for s, a in zip(test_df["sentence"], test_df["aspect"])
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_test, vectorizer


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train, X_test, vectorizer = build_features(train_df, test_df)
    y_train, y_test = train_df["polarity_label"], test_df["polarity_label"]

    results = {}

    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ("Linear SVM", LinearSVC(class_weight="balanced", C=1.0)),
    ]:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, digits=3)
        macro_f1 = f1_score(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds, labels=["negative", "neutral", "positive"])
        results[name] = {
            "model": clf,
            "preds": preds,
            "report": report,
            "macro_f1": macro_f1,
            "confusion_matrix": cm,
        }
        print(f"\n===== {name} =====")
        print(report)
        print("Macro F1:", round(macro_f1, 4))

    return results, vectorizer


def plot_confusion_matrix(cm, labels, title, out_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    import os

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    results, vectorizer = train_and_evaluate(train_df, test_df)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best = results[best_name]
    plot_confusion_matrix(
        best["confusion_matrix"], ["negative", "neutral", "positive"],
        f"Confusion Matrix - {best_name}", "figures/confusion_matrix_classical.png",
    )

    with open("reports/classical_model_report.txt", "w") as f:
        for name, res in results.items():
            f.write(f"===== {name} =====\n{res['report']}\nMacro F1: {res['macro_f1']:.4f}\n\n")

    print(f"\nBest classical model: {best_name} (Macro F1={best['macro_f1']:.4f})")
