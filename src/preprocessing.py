"""
preprocessing.py
------------------
Light-weight text cleaning for the ABSA pipeline. Deliberately does NOT
remove stopwords or aggressively stem, because negation words and
short function words ("not", "n't", "but") carry sentiment signal that
matters a lot for aspect-based sentiment.
"""

import re


def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9$'.,!? ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_aspect_aware_input(sentence: str, aspect: str) -> str:
    """Builds the '[aspect] [SEP] [sentence]' style input used for both
    the classical TF-IDF baseline and the transformer model. Putting the
    aspect term first lets a bag-of-words model weight aspect-adjacent
    words more naturally when combined with n-grams, and matches the
    sentence-pair format expected by BERT/DeBERTa in the transformer
    stage (see notebooks/03_transformer_model.ipynb).
    """
    return f"{aspect} [SEP] {sentence}"
