"""
Email Spam Classifier
=====================
A machine learning pipeline for classifying emails as spam or ham.
Supports multiple models: Naive Bayes, Logistic Regression, SVM, Random Forest.
"""

import re
import string
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Text Preprocessing
# ──────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalize email text.
    Steps: lowercase → remove URLs → remove email addresses →
           remove numbers → remove punctuation → collapse whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)          # URLs → token
    text = re.sub(r"\S+@\S+", " email ", text)                # emails → token
    text = re.sub(r"\b\d+\b", " num ", text)                  # numbers → token
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset. Supports CSV files with columns: 'text', 'label'
    where label is 'spam'/'ham' or 1/0.
    """
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Handle common dataset formats (e.g. SMS Spam Collection)
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    assert "text" in df.columns, "Dataset must have a 'text' column."
    assert "label" in df.columns, "Dataset must have a 'label' column."

    # Encode labels
    df["label"] = df["label"].map({"spam": 1, "ham": 0, 1: 1, 0: 0})
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(preprocess_text)

    print(f"✅ Loaded {len(df)} samples | Spam: {df['label'].sum()} | Ham: {(df['label']==0).sum()}")
    return df


def load_sample_data() -> pd.DataFrame:
    """Generate a small synthetic dataset for quick testing."""
    spam_samples = [
        "Congratulations! You've won a $1000 gift card. Click here now!",
        "FREE pills! Lose 30 lbs in 30 days. No prescription needed.",
        "URGENT: Your account has been compromised. Verify now at http://phish.com",
        "You are selected for a lottery prize. Send your bank details.",
        "Buy cheap Viagra online. No doctor needed. Discreet shipping!",
        "Make $5000 a week working from home. Guaranteed! Limited offer.",
        "Hot singles in your area are waiting for you. Click now!",
        "Your PayPal account is suspended. Update your information immediately.",
        "Claim your free iPhone 15. Hurry! Only 10 left in stock.",
        "Earn $500/day doing nothing. Work at home opportunity. Apply now!",
    ]
    ham_samples = [
        "Hi John, just wanted to confirm our meeting tomorrow at 10am.",
        "Please find the attached quarterly report for your review.",
        "Can we reschedule the call to Thursday? I have a conflict.",
        "Happy birthday! Hope you have a wonderful day.",
        "The project deadline has been moved to next Friday.",
        "Thanks for your help yesterday. I really appreciate it.",
        "Can you send me the updated budget spreadsheet when you get a chance?",
        "Reminder: Team lunch is at noon in the main conference room.",
        "I reviewed your proposal and have some feedback to share.",
        "Looking forward to seeing you at the conference next week.",
    ]

    texts = spam_samples + ham_samples
    labels = [1] * len(spam_samples) + [0] * len(ham_samples)

    df = pd.DataFrame({"text": texts, "label": labels})
    df["text"] = df["text"].apply(preprocess_text)
    return df


# ──────────────────────────────────────────────
# Model Building
# ──────────────────────────────────────────────

def build_pipelines() -> dict:
    """
    Build sklearn Pipelines for each model.
    Each pipeline: TF-IDF Vectorizer → Classifier.
    """
    tfidf_params = dict(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )

    pipelines = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
    }
    return pipelines


# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute and display evaluation metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Ham','Spam'])}")

    return metrics


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Split data, train all models, evaluate and return the best one.
    Returns (best_model, results_df, X_test, y_test).
    """
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n📊 Train size: {len(X_train)} | Test size: {len(X_test)}")

    pipelines = build_pipelines()
    all_metrics = []
    trained_models = {}

    for name, pipeline in pipelines.items():
        print(f"\n⏳ Training {name}...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics).set_index("model")
    best_model_name = results_df["f1"].idxmax()
    best_model = trained_models[best_model_name]

    print(f"\n🏆 Best Model: {best_model_name} (F1 = {results_df.loc[best_model_name, 'f1']:.4f})")
    return best_model, results_df, X_test, y_test


# ──────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────

def save_model(model, path: str = "models/spam_classifier.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved → {path}")


def load_model(path: str = "models/spam_classifier.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"📂 Model loaded ← {path}")
    return model


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict(model, texts: list) -> list:
    """
    Predict spam (1) or ham (0) for a list of raw email strings.
    Returns list of dicts with label and confidence.
    """
    cleaned = [preprocess_text(t) for t in texts]
    preds = model.predict(cleaned)

    # Confidence: use predict_proba if available, else decision_function
    results = []
    for text, pred in zip(texts, preds):
        label = "SPAM 🚨" if pred == 1 else "HAM ✅"
        results.append({"text": text[:80] + "...", "prediction": label})
    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("📧 Email Spam Classifier — Training Pipeline\n")

    # Use sample data (replace with load_data("data/emails.csv") for real data)
    df = load_sample_data()

    best_model, results, X_test, y_test = train_and_evaluate(df)

    save_model(best_model)

    # Demo predictions
    test_emails = [
        "Congratulations! You've won a FREE iPhone. Click here to claim your prize!",
        "Hi, just checking if you're available for a call tomorrow afternoon.",
    ]
    print("\n🔍 Demo Predictions:")
    for r in predict(best_model, test_emails):
        print(f"  [{r['prediction']}] {r['text']}")
