# Sentiment Analysis and Aspect-Based Opinion Mining of Customer Reviews for Data-Driven Business Insights using NLP

A college NLP project that goes beyond overall "positive/negative" review
sentiment and predicts sentiment **per aspect** (food, service, price,
ambience, ...) of a customer review, then rolls those predictions up into a
business-facing insights dashboard — the kind of output a real Data
Analyst would hand to a restaurant/product manager.

> But the **staff** was so horrible to us → *staff: negative*
> The **food** is uniformly exceptional → *food: positive*

## Why aspect-based, not just overall sentiment?

A single review often contains mixed sentiment about different things.
Aspect-Based Sentiment Analysis (ABSA) recovers that nuance, which is what
makes the output business-actionable ("service is the problem, not food")
rather than a single vague sentiment score per review.

## Dataset

[SemEval-2014 Task 4](http://alt.qcri.org/semeval2014/task4/) — Restaurant
Reviews, the standard academic benchmark for ABSA. 3,608 train / 1,120 test
labeled (sentence, aspect term, polarity) examples.

## Reference papers

- *Instruct-DeBERTa: A Hybrid Approach for Aspect-Based Sentiment Analysis* — [arXiv:2408.13202](https://arxiv.org/abs/2408.13202)
- Cross-domain aspect-based sentiment analysis (IIT Guwahati) — [arXiv:2501.08974](https://arxiv.org/abs/2501.08974)

## Project structure

```
├── data/
│   ├── raw/                     # original SemEval .xml.seg files
│   └── processed/                # parsed train.csv / test.csv
├── src/
│   ├── data_utils.py              # parse raw SemEval format -> DataFrame
│   ├── preprocessing.py           # text cleaning + aspect-aware input builder
│   ├── aspect_categories.py       # rolls ~100s of raw aspect terms into 6 business categories
│   ├── classical_model.py         # TF-IDF + Logistic Regression / Linear SVM baseline
│   ├── transformer_model.py       # fine-tunes DistilBERT/DeBERTa (run in Colab, needs GPU)
│   ├── business_insights.py       # aggregates predictions into category-level KPIs + charts
│   └── eda.py                     # exploratory data analysis charts
├── notebooks/
│   └── ABSA_Business_Insights.ipynb   # full narrated pipeline, start to finish
├── figures/                        # generated charts (PNG)
├── reports/                         # generated summary tables
├── requirements.txt
└── README.md
```

## Pipeline

1. **Data loading & parsing** — raw SemEval format → clean DataFrame
2. **EDA** — class balance, sentence length, most frequent aspect terms
3. **Preprocessing** — light cleaning, aspect-aware input (`"<aspect> [SEP] <sentence>"`)
4. **Baseline model** — TF-IDF (uni+bigrams) + Logistic Regression / Linear SVM
5. **Transformer model** — fine-tuned DistilBERT (or swap in `deberta-v3-base`), run separately in Colab
6. **Business insights** — roll aspect-level predictions up into 6 categories (Food Quality, Service, Price/Value, Ambience, Drinks, Location/Wait), compute mention volume, sentiment mix, and a net sentiment score (`%positive − %negative`) per category
7. **Conclusions & limitations**

## Results (held-out test set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.704 | 0.590 |
| Linear SVM (TF-IDF) | 0.709 | 0.577 |
| DistilBERT (fine-tuned, 3 epochs, Colab T4 GPU) | **0.814** | **0.704** |

The classical baselines do well on the majority `positive` class (F1 ≈ 0.83)
but struggle on `neutral` (F1 ≈ 0.35–0.38) — a known hard case in ABSA.
Fine-tuning DistilBERT closes much of that gap, lifting macro-F1 from ~0.59
to 0.704, confirming that contextual embeddings handle negation and
aspect-specific context better than TF-IDF bag-of-words features. There's
still headroom before matching published DeBERTa-based SOTA on this
benchmark (~82–86% accuracy / ~75–80 macro-F1) — see Limitations.

## How to run

**Classical pipeline (no GPU needed):**
```bash
pip install -r requirements.txt
python src/data_utils.py         # parses raw data -> data/processed/*.csv
python src/eda.py                # -> figures/
python src/classical_model.py    # -> reports/classical_model_report.txt, figures/confusion_matrix_classical.png
python src/business_insights.py  # -> reports/business_insights_summary.csv, figures/*.png
jupyter notebook notebooks/ABSA_Business_Insights.ipynb
```

**Transformer pipeline (Google Colab, free T4 GPU):**
```bash
!pip install -q transformers datasets evaluate accelerate
!python src/transformer_model.py --model_name distilbert-base-uncased --epochs 3
```

## Limitations & possible extensions

- Uses SemEval's *given* aspect terms — aspect term extraction itself
  (finding which words are the aspects) is a separate, harder subtask not
  attempted here.
- The aspect→category taxonomy is a manually built keyword map; it won't
  generalize perfectly beyond restaurant-domain vocabulary.
- Natural next steps: full end-to-end aspect extraction + sentiment,
  a small demo UI (Streamlit/Gradio) highlighting per-aspect sentiment
  inline in a pasted review, or applying the pipeline to a larger, real
  review corpus (e.g. Yelp Open Dataset).

## Author

Akil — B.Tech Artificial Intelligence, Mumbai
