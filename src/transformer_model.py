"""
transformer_model.py
-----------------------
Fine-tunes a pretrained transformer (DistilBERT by default, swap in
'microsoft/deberta-v3-base' for closer alignment with the Instruct-DeBERTa
reference paper if you have a bigger GPU/more time) for aspect-based
sentiment classification on SemEval-2014 Restaurants.

Input format per example: (aspect, sentence) sentence-pair -> one of
{negative, neutral, positive}. This mirrors the standard ABSA
formulation used in both reference papers for this project:
  - Instruct-DeBERTa (arXiv:2408.13202)
  - IIT Guwahati cross-domain ABSA paper (arXiv:2501.08974)

NOTE: this script needs internet access to Hugging Face Hub and,
ideally, a GPU. Run it in Google Colab (free T4 GPU is enough) or
Kaggle Notebooks -- it is NOT meant to run inside a restricted/offline
sandbox. In Colab:

    !pip install -q transformers datasets evaluate accelerate
    !python transformer_model.py

Usage:
    python src/transformer_model.py --model_name distilbert-base-uncased --epochs 3
"""

import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    train_df["label"] = train_df["polarity_label"].map(LABEL2ID)
    test_df["label"] = test_df["polarity_label"].map(LABEL2ID)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="models/transformer_absa")
    args = parser.parse_args()

    train_ds, test_ds = load_data()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        # sentence-pair input: (aspect, sentence) -> lets the model attend
        # to the aspect term as a separate segment rather than just
        # concatenated text.
        return tokenizer(
            batch["aspect"], batch["sentence"],
            truncation=True, padding="max_length", max_length=96,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final test metrics:", metrics)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
