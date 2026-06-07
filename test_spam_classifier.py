"""
tests/test_spam_classifier.py
Unit tests for the Email Spam Classifier.
Run with: pytest tests/ -v
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from spam_classifier import (
    preprocess_text, load_sample_data,
    build_pipelines, train_and_evaluate, predict
)


# ──────────────────────────────────────────────
# Preprocessing Tests
# ──────────────────────────────────────────────

class TestPreprocessText:
    def test_lowercases(self):
        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        result = preprocess_text("Visit http://spam.com now!")
        assert "url" in result
        assert "http" not in result

    def test_removes_email_addresses(self):
        result = preprocess_text("Contact us at spam@spam.com")
        assert "email" in result

    def test_replaces_numbers(self):
        result = preprocess_text("Win $1000 now")
        assert "num" in result

    def test_removes_punctuation(self):
        result = preprocess_text("Hello, world!!!")
        assert "," not in result
        assert "!" not in result

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_whitespace_collapsed(self):
        result = preprocess_text("hello   world")
        assert "  " not in result


# ──────────────────────────────────────────────
# Data Loading Tests
# ──────────────────────────────────────────────

class TestLoadSampleData:
    def test_returns_dataframe(self):
        df = load_sample_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = load_sample_data()
        assert "text" in df.columns
        assert "label" in df.columns

    def test_labels_are_binary(self):
        df = load_sample_data()
        assert set(df["label"].unique()).issubset({0, 1})

    def test_has_both_classes(self):
        df = load_sample_data()
        assert 0 in df["label"].values
        assert 1 in df["label"].values

    def test_no_null_values(self):
        df = load_sample_data()
        assert df["text"].isnull().sum() == 0
        assert df["label"].isnull().sum() == 0


# ──────────────────────────────────────────────
# Pipeline Tests
# ──────────────────────────────────────────────

class TestBuildPipelines:
    def test_returns_dict(self):
        pipelines = build_pipelines()
        assert isinstance(pipelines, dict)

    def test_has_expected_models(self):
        pipelines = build_pipelines()
        expected = {"Naive Bayes", "Logistic Regression", "SVM", "Random Forest"}
        assert set(pipelines.keys()) == expected

    def test_pipelines_have_fit_predict(self):
        for name, pipe in build_pipelines().items():
            assert hasattr(pipe, "fit")
            assert hasattr(pipe, "predict")


# ──────────────────────────────────────────────
# Training & Evaluation Tests
# ──────────────────────────────────────────────

class TestTrainAndEvaluate:
    @pytest.fixture
    def sample_df(self):
        return load_sample_data()

    def test_returns_best_model(self, sample_df):
        model, results, X_test, y_test = train_and_evaluate(sample_df)
        assert model is not None

    def test_results_has_all_metrics(self, sample_df):
        _, results, _, _ = train_and_evaluate(sample_df)
        for col in ["accuracy", "precision", "recall", "f1"]:
            assert col in results.columns

    def test_metrics_in_valid_range(self, sample_df):
        _, results, _, _ = train_and_evaluate(sample_df)
        for col in ["accuracy", "precision", "recall", "f1"]:
            assert (results[col] >= 0).all()
            assert (results[col] <= 1).all()


# ──────────────────────────────────────────────
# Prediction Tests
# ──────────────────────────────────────────────

class TestPredict:
    @pytest.fixture
    def trained_model(self):
        df = load_sample_data()
        model, _, _, _ = train_and_evaluate(df)
        return model

    def test_returns_list(self, trained_model):
        results = predict(trained_model, ["Hello world"])
        assert isinstance(results, list)

    def test_result_has_keys(self, trained_model):
        results = predict(trained_model, ["Hello world"])
        assert "prediction" in results[0]
        assert "text" in results[0]

    def test_obvious_spam_detected(self, trained_model):
        spam = ["Congratulations! You won FREE money! Click here now!"]
        results = predict(trained_model, spam)
        assert "SPAM" in results[0]["prediction"]

    def test_obvious_ham_detected(self, trained_model):
        ham = ["Hi, can we schedule a meeting tomorrow?"]
        results = predict(trained_model, ham)
        assert "HAM" in results[0]["prediction"]

    def test_multiple_emails(self, trained_model):
        emails = ["Win a prize now!", "See you tomorrow", "Free offer!"]
        results = predict(trained_model, emails)
        assert len(results) == 3
