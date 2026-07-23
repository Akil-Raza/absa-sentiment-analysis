"""
data_utils.py
--------------
Utilities to parse the SemEval-2014 Task 4 (Restaurants) aspect-based
sentiment analysis dataset into a clean pandas DataFrame.

The raw ``.xml.seg`` format stores each labeled example as 3 lines:
    line 1: sentence text, with the aspect term replaced by the placeholder $T$
    line 2: the aspect term itself
    line 3: polarity label as an integer: -1 (negative), 0 (neutral), 1 (positive)

Example:
    But the $T$ was so horrible to us .
    staff
    -1
"""

import re
import pandas as pd

POLARITY_MAP = {-1: "negative", 0: "neutral", 1: "positive"}


def parse_absa_file(path: str) -> pd.DataFrame:
    """Parse a SemEval .xml.seg file into a DataFrame with columns:
    ['sentence_masked', 'sentence', 'aspect', 'polarity', 'polarity_label']
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip("\n") for l in f.readlines()]

    records = []
    for i in range(0, len(lines) - 2, 3):
        masked_sentence = lines[i].strip()
        aspect = lines[i + 1].strip()
        polarity = int(lines[i + 2].strip())
        full_sentence = masked_sentence.replace("$T$", aspect)
        full_sentence = re.sub(r"\s+", " ", full_sentence).strip()
        records.append(
            {
                "sentence_masked": masked_sentence,
                "sentence": full_sentence,
                "aspect": aspect,
                "polarity": polarity,
                "polarity_label": POLARITY_MAP[polarity],
            }
        )
    return pd.DataFrame(records)


def load_train_test(train_path: str, test_path: str):
    """Convenience loader returning (train_df, test_df)."""
    train_df = parse_absa_file(train_path)
    test_df = parse_absa_file(test_path)
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_train_test(
        "data/raw/Restaurants_Train.xml.seg",
        "data/raw/Restaurants_Test_Gold.xml.seg",
    )
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(train_df.head())
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
