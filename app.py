"""
app.py
------
Interactive dashboard for the ABSA (Aspect-Based Sentiment Analysis)
project. Trains the same classical pipeline as src/classical_model.py
and src/business_insights.py on startup (a few seconds, cached), then
exposes it as a browsable, filterable, and live-testable dashboard.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud by pointing it at this
file in your GitHub repo — see the README for the two-minute setup.
"""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

st.set_page_config(page_title="ABSA Insights", page_icon="🍽️", layout="wide")

DATA_DIR = Path(__file__).parent / "data" / "processed"

SENTIMENT_COLORS = {"positive": "#2F7D4F", "neutral": "#96792F", "negative": "#B23A2E"}
SENTIMENT_ORDER = ["positive", "neutral", "negative"]

# --------------------------------------------------------------------------
# Preprocessing — mirrors src/preprocessing.py exactly, so results here
# match the numbers in the README / reports/ folder.
# --------------------------------------------------------------------------


def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9$'.,!? ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_aspect_aware_input(sentence: str, aspect: str) -> str:
    return f"{aspect} [SEP] {sentence}"


# --------------------------------------------------------------------------
# Aspect -> business category taxonomy — mirrors src/aspect_categories.py
# --------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "Food Quality": [
        "food", "dish", "meal", "menu", "taste", "flavor", "flavour",
        "sushi", "pizza", "sauce", "meat", "fish", "chicken", "steak",
        "dessert", "appetizer", "portion", "ingredient", "spicy",
        "cuisine", "curry", "noodle", "rice", "bread", "cook",
        "lunch", "dinner", "brunch", "entree", "salad", "burger",
        "soup", "sandwich", "seafood", "pasta", "crust", "cake",
        "selection",
    ],
    "Service": [
        "service", "staff", "waiter", "waitress", "server", "host",
        "hostess", "manager", "attitude", "attentive", "rude",
        "friendly", "wait time", "reservation", "bartender", "owner",
        "serve", "waitstaff", "waiting",
    ],
    "Price/Value": [
        "price", "cost", "value", "expensive", "cheap", "bill",
        "money", "worth", "overpriced", "affordable",
    ],
    "Ambience": [
        "ambiance", "ambience", "atmosphere", "decor", "interior",
        "music", "noise", "seating", "space", "view", "lighting",
        "vibe", "crowd", "romantic", "place", "dining experience",
        "decoration",
    ],
    "Drinks": [
        "wine", "cocktail", "beer", "drink", "beverage", "bar",
        "margarita", "sake", "coffee", "tea",
    ],
    "Location/Wait": [
        "location", "parking", "wait", "line", "queue", "crowded",
        "delivery", "takeout", "speed",
    ],
}


def categorize_aspect(aspect: str) -> str:
    a = aspect.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in a:
                return category
    return "Other"


# --------------------------------------------------------------------------
# Data + model (cached so the app only trains once per session/restart)
# --------------------------------------------------------------------------


@st.cache_data
def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


@st.cache_resource
def train_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_text = [basic_clean(make_aspect_aware_input(s, a)) for s, a in zip(train_df.sentence, train_df.aspect)]
    test_text = [basic_clean(make_aspect_aware_input(s, a)) for s, a in zip(test_df.sentence, test_df.aspect)]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    y_train, y_test = train_df["polarity_label"], test_df["polarity_label"]

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X_train, y_train)
    svm = LinearSVC(class_weight="balanced", C=1.0).fit(X_train, y_train)

    preds_lr = logreg.predict(X_test)
    preds_svm = svm.predict(X_test)

    metrics = {
        "Logistic Regression": {
            "accuracy": accuracy_score(y_test, preds_lr),
            "macro_f1": f1_score(y_test, preds_lr, average="macro"),
        },
        "Linear SVM": {
            "accuracy": accuracy_score(y_test, preds_svm),
            "macro_f1": f1_score(y_test, preds_svm, average="macro"),
        },
        # Trained separately on a Colab GPU — see src/transformer_model.py and the README.
        "DistilBERT (fine-tuned)": {"accuracy": 0.814, "macro_f1": 0.704},
    }
    cm = confusion_matrix(y_test, preds_lr, labels=SENTIMENT_ORDER)
    return vectorizer, logreg, preds_lr, metrics, cm


def build_business_summary(test_df: pd.DataFrame, preds) -> pd.DataFrame:
    df = test_df.copy()
    df["aspect_category"] = df["aspect"].apply(categorize_aspect)
    df["pred"] = preds
    df["correct"] = df["pred"] == df["polarity_label"]

    grouped = df.groupby("aspect_category")
    summary = grouped.agg(
        mentions=("polarity_label", "count"),
        pct_positive=("polarity_label", lambda s: round((s == "positive").mean() * 100, 1)),
        pct_neutral=("polarity_label", lambda s: round((s == "neutral").mean() * 100, 1)),
        pct_negative=("polarity_label", lambda s: round((s == "negative").mean() * 100, 1)),
    ).reset_index()

    def net_score(s):
        counts = s.value_counts(normalize=True) * 100
        return round(counts.get("positive", 0.0) - counts.get("negative", 0.0), 1)

    summary["net_sentiment_score"] = grouped["polarity_label"].apply(net_score).values
    summary = summary.sort_values("mentions", ascending=False).reset_index(drop=True)
    return summary, df


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

train_df, test_df = load_data()
vectorizer, logreg, preds_lr, metrics, cm = train_models(train_df, test_df)
summary_df, explorer_df = build_business_summary(test_df, preds_lr)

st.title("🍽️ ABSA Insights")
st.caption(
    "Aspect-based sentiment analysis on the SemEval-2014 Restaurant Reviews benchmark — "
    "predicting sentiment *per aspect* of a review, not just overall."
)

tab_overview, tab_categories, tab_models, tab_try, tab_explorer = st.tabs(
    ["Overview", "Categories", "Models", "Try it yourself", "Explorer"]
)

# ---------------- Overview ----------------
with tab_overview:
    best_row = summary_df.loc[summary_df["net_sentiment_score"].idxmax()]
    eligible = summary_df[summary_df["mentions"] >= 20]
    worst_row = eligible.loc[eligible["net_sentiment_score"].idxmin()]
    best_model_name = max(metrics, key=lambda k: metrics[k]["accuracy"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labeled mentions (train + test)", f"{len(train_df) + len(test_df):,}")
    c2.metric("Strongest category", best_row["aspect_category"], f"net +{best_row['net_sentiment_score']}")
    c3.metric("Weakest category (≥20 mentions)", worst_row["aspect_category"], f"net {worst_row['net_sentiment_score']:+}")
    c4.metric(f"Best model: {best_model_name}", f"{metrics[best_model_name]['accuracy'] * 100:.1f}% accuracy")

    st.markdown("#### Why aspect-based, not just overall sentiment?")
    st.markdown(
        "> But the **staff** was so horrible to us → *staff: negative*\n>\n"
        "> The **food** is uniformly exceptional → *food: positive*\n\n"
        "A single review often carries mixed sentiment about different things. "
        "Scoring it as one number throws that nuance away — ABSA keeps it, "
        "which is what makes the rollup on the **Categories** tab business-actionable."
    )
    st.info(
        f"Service has the highest negative-mention share of any high-volume category "
        f"({summary_df.loc[summary_df.aspect_category == 'Service', 'pct_negative'].iloc[0]}%) "
        f"despite a positive net score — worth a closer look on the Categories tab.",
        icon="💡",
    )

# ---------------- Categories ----------------
with tab_categories:
    st.markdown("1,120 test-set aspect mentions rolled up into six business categories plus a catch-all.")
    metric_choice = st.radio(
        "Metric", ["Mentions", "Net sentiment score", "Sentiment mix"], horizontal=True, label_visibility="collapsed"
    )

    ordered = summary_df.sort_values("mentions", ascending=True)

    if metric_choice == "Mentions":
        fig = px.bar(
            ordered, x="mentions", y="aspect_category", orientation="h",
            color_discrete_sequence=["#1F5C58"], text="mentions",
        )
        fig.update_layout(xaxis_title="Number of mentions", yaxis_title="")
    elif metric_choice == "Net sentiment score":
        ordered = ordered.sort_values("net_sentiment_score")
        colors = ["#B23A2E" if v < 0 else "#2F7D4F" for v in ordered["net_sentiment_score"]]
        fig = go.Figure(go.Bar(
            x=ordered["net_sentiment_score"], y=ordered["aspect_category"],
            orientation="h", marker_color=colors, text=ordered["net_sentiment_score"],
        ))
        fig.update_layout(xaxis_title="Net sentiment score (%positive − %negative)", yaxis_title="")
        fig.add_vline(x=0, line_color="gray", line_width=1)
    else:
        long_df = ordered.melt(
            id_vars="aspect_category",
            value_vars=["pct_positive", "pct_neutral", "pct_negative"],
            var_name="sentiment", value_name="pct",
        )
        long_df["sentiment"] = long_df["sentiment"].str.replace("pct_", "")
        fig = px.bar(
            long_df, x="pct", y="aspect_category", color="sentiment", orientation="h",
            color_discrete_map=SENTIMENT_COLORS, category_orders={"sentiment": SENTIMENT_ORDER},
        )
        fig.update_layout(xaxis_title="Percentage of mentions", yaxis_title="", barmode="stack")

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width='stretch')

# ---------------- Models ----------------
with tab_models:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("##### Accuracy & macro F1 by model")
        model_df = pd.DataFrame(
            [{"model": k, "Accuracy": v["accuracy"] * 100, "Macro F1": v["macro_f1"] * 100} for k, v in metrics.items()]
        ).melt(id_vars="model", var_name="metric", value_name="score")
        fig2 = px.bar(
            model_df, x="model", y="score", color="metric", barmode="group",
            color_discrete_sequence=["#1F5C58", "#96792F"],
        )
        fig2.update_layout(yaxis_title="Score (%)", xaxis_title="", height=360, plot_bgcolor="white")
        st.plotly_chart(fig2, width='stretch')

    with right:
        st.markdown("##### Confusion matrix — Logistic Regression")
        fig3 = px.imshow(
            cm, x=SENTIMENT_ORDER, y=SENTIMENT_ORDER, text_auto=True,
            color_continuous_scale="Greens", labels=dict(x="Predicted", y="True", color="Count"),
        )
        fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, width='stretch')

    st.caption(
        "Classical models do well on the majority *positive* class (F1 ≈ 0.83) but struggle on "
        "*neutral* (F1 ≈ 0.35–0.38) — a known hard case in ABSA. DistilBERT was fine-tuned "
        "separately on a Colab T4 GPU (see `src/transformer_model.py`)."
    )

# ---------------- Try it yourself ----------------
with tab_try:
    st.markdown("Runs the actual trained TF-IDF + Logistic Regression model above — live, on whatever you type.")

    examples = {
        "negative": explorer_df[(explorer_df["polarity_label"] == "negative") & (explorer_df["sentence"].str.len().between(30, 90))],
        "neutral": explorer_df[(explorer_df["polarity_label"] == "neutral") & (explorer_df["sentence"].str.len().between(30, 90))],
        "positive": explorer_df[(explorer_df["polarity_label"] == "positive") & (explorer_df["sentence"].str.len().between(30, 90))],
    }

    if "aspect_input" not in st.session_state:
        st.session_state.aspect_input = "service"
        st.session_state.sentence_input = "The staff was rude but the food made up for it."

    cols = st.columns(3)
    for col, (label, subset) in zip(cols, examples.items()):
        if len(subset) and col.button(f"Try a {label} example"):
            row = subset.iloc[0]
            st.session_state.aspect_input = row["aspect"]
            st.session_state.sentence_input = row["sentence"]

    left, right = st.columns(2)
    with left:
        aspect = st.text_input("Aspect term", key="aspect_input")
        sentence = st.text_area("Review sentence", key="sentence_input", height=100)
        run = st.button("Predict sentiment", type="primary")

    with right:
        if run and aspect and sentence:
            cleaned = basic_clean(make_aspect_aware_input(sentence, aspect))
            vec = vectorizer.transform([cleaned])
            proba = logreg.predict_proba(vec)[0]
            classes = logreg.classes_
            top_idx = proba.argmax()
            top_label = classes[top_idx]

            color_map = {"positive": "green", "neutral": "orange", "negative": "red"}
            st.markdown(f"**Predicted: :{color_map[top_label]}[{top_label}]**")
            for label in SENTIMENT_ORDER:
                p = proba[list(classes).index(label)]
                st.write(f"{label} — {p*100:.1f}%")
                st.progress(float(p))
        else:
            st.caption("Enter an aspect and a sentence, or tap an example above, then hit predict.")

# ---------------- Explorer ----------------
with tab_explorer:
    st.markdown(f"All {len(explorer_df)} held-out test-set predictions the model was scored on.")

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    query = f1.text_input("Search sentence or aspect")
    cat_filter = f2.multiselect("Category", sorted(explorer_df["aspect_category"].unique()))
    sent_filter = f3.multiselect("True label", SENTIMENT_ORDER)
    correct_filter = f4.selectbox("Predictions", ["All", "Correct only", "Incorrect only"])

    filtered = explorer_df.copy()
    if query:
        mask = filtered["sentence"].str.contains(query, case=False, na=False) | filtered["aspect"].str.contains(query, case=False, na=False)
        filtered = filtered[mask]
    if cat_filter:
        filtered = filtered[filtered["aspect_category"].isin(cat_filter)]
    if sent_filter:
        filtered = filtered[filtered["polarity_label"].isin(sent_filter)]
    if correct_filter == "Correct only":
        filtered = filtered[filtered["correct"]]
    elif correct_filter == "Incorrect only":
        filtered = filtered[~filtered["correct"]]

    st.caption(f"{len(filtered)} of {len(explorer_df)} rows")
    st.dataframe(
        filtered[["sentence", "aspect", "aspect_category", "polarity_label", "pred", "correct"]].rename(
            columns={"aspect_category": "category", "polarity_label": "true", "pred": "predicted"}
        ),
        width='stretch',
        height=460,
    )

st.divider()
st.caption(
    "Dataset: SemEval-2014 Task 4, Restaurant Reviews · "
    "[absa-sentiment-analysis on GitHub](https://github.com/Akil-Raza/absa-sentiment-analysis)"
)