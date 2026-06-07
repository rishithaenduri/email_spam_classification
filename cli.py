"""
cli.py — Command-line interface for the Email Spam Classifier.

Usage:
    python cli.py train --data data/emails.csv
    python cli.py predict --email "You won a free prize!"
    python cli.py predict --file data/test_emails.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spam_classifier import (
    load_data, load_sample_data, train_and_evaluate,
    save_model, load_model, predict, preprocess_text
)

MODEL_PATH = "models/spam_classifier.pkl"


def cmd_train(args):
    if args.data:
        df = load_data(args.data)
    else:
        print("⚠️  No dataset provided. Using built-in sample data.")
        df = load_sample_data()

    best_model, results, _, _ = train_and_evaluate(df, test_size=args.test_size)
    save_model(best_model, MODEL_PATH)

    print("\n📈 Model Comparison:")
    print(results.round(4).to_string())


def cmd_predict(args):
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        print("❌ No trained model found. Run: python cli.py train")
        sys.exit(1)

    emails = []
    if args.email:
        emails = [args.email]
    elif args.file:
        with open(args.file) as f:
            emails = [line.strip() for line in f if line.strip()]
    else:
        print("❌ Provide --email or --file.")
        sys.exit(1)

    results = predict(model, emails)
    print("\n🔍 Predictions:")
    for r in results:
        print(f"  {r['prediction']}  →  {r['text']}")


def main():
    parser = argparse.ArgumentParser(
        description="📧 Email Spam Classifier CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train on your dataset:
    python cli.py train --data data/emails.csv

  Train on sample data:
    python cli.py train

  Predict a single email:
    python cli.py predict --email "You won a free iPhone!"

  Predict from a file:
    python cli.py predict --file data/test_emails.txt
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the classifier")
    train_parser.add_argument("--data", type=str, help="Path to CSV dataset (optional)")
    train_parser.add_argument("--test-size", type=float, default=0.2,
                              help="Fraction for test set (default: 0.2)")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Classify emails")
    group = predict_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--email", type=str, help="Single email text to classify")
    group.add_argument("--file", type=str, help="Text file with one email per line")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
