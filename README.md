# 📧 Email Spam Classifier

A machine learning project that classifies emails as **spam** or **ham** using Natural Language Processing and multiple ML models.

---

## 🔍 Project Overview

This project builds a complete spam detection pipeline:

- **Text preprocessing**: URL/email/number normalization, lowercasing, punctuation removal
- **Feature extraction**: TF-IDF with unigrams and bigrams
- **Multiple ML models**: Naive Bayes, Logistic Regression, SVM, Random Forest
- **Automatic model selection**: Best model by F1 score
- **CLI tool**: Train and predict from the command line
- **Unit tests**: Full test suite with `pytest`

---

## 📁 Project Structure

```
email-spam-classifier/
│
├── src/
│   ├── spam_classifier.py   # Core ML pipeline (preprocessing, training, evaluation)
│   └── cli.py               # Command-line interface
│
├── notebooks/
│   └── spam_classifier_notebook.ipynb  # EDA + model exploration
│
├── tests/
│   └── test_spam_classifier.py         # Unit tests (pytest)
│
├── data/                    # Put your dataset here
├── models/                  # Trained model saved here
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/email-spam-classifier.git
cd email-spam-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
# Using built-in sample data
python src/cli.py train

# Using your own dataset (CSV with 'text' and 'label' columns)
python src/cli.py train --data data/emails.csv
```

### 4. Predict

```bash
# Single email
python src/cli.py predict --email "Congratulations! You've won a free iPhone!"

# From a file (one email per line)
python src/cli.py predict --file data/test_emails.txt
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## 📊 Models & Performance

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| Naive Bayes         | ~0.95    | ~0.96     | ~0.93  | ~0.94  |
| Logistic Regression | ~0.97    | ~0.97     | ~0.96  | ~0.96  |
| SVM (LinearSVC)     | ~0.98    | ~0.98     | ~0.97  | ~0.97  |
| Random Forest       | ~0.96    | ~0.96     | ~0.95  | ~0.95  |

> Metrics vary by dataset. Train on the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) for real benchmarks.

---

## 📦 Dataset

You can use any CSV file with these columns:

| Column  | Values              |
|---------|---------------------|
| `text`  | Raw email/SMS text  |
| `label` | `spam` / `ham` or `1` / `0` |

**Recommended public datasets:**
- [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Enron Spam Dataset](https://www2.aueb.gr/users/ion/data/enron-spam/)

---

## 🛠️ Tech Stack

| Tool         | Purpose                        |
|--------------|-------------------------------|
| Python 3.10+ | Language                       |
| scikit-learn | ML models & pipelines          |
| pandas       | Data handling                  |
| numpy        | Numerical operations           |
| matplotlib   | Visualizations                 |
| seaborn      | Statistical plots              |
| wordcloud    | Word frequency visualization   |
| pytest       | Unit testing                   |

---

## 🧪 How It Works

```
Raw Email Text
      │
      ▼
  Preprocessing
  (lowercase, remove URLs/numbers/punctuation)
      │
      ▼
  TF-IDF Vectorization
  (unigrams + bigrams, top 10,000 features)
      │
      ▼
  ML Classifier
  (NB / LR / SVM / RF)
      │
      ▼
  Spam ✅ or Ham 🚨
```

---

## 👤 Author

**Vithey** — Internship Project  
Feel free to fork, star ⭐, and contribute!

---

## 📄 License

MIT License. See `LICENSE` for details.
