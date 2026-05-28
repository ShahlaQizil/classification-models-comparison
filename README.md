# Classification Models — Comparison & Evaluation

A machine learning project that implements and compares three classic classification
algorithms — Logistic Regression, Decision Tree, and SVM — evaluating each with
accuracy, precision, recall, and F1 score. The Wisconsin Breast Cancer dataset is
used as the working example.

## Overview

This script walks through a complete supervised-learning workflow:

1. **Load and explore** the dataset
2. **Preprocess** — handle missing values and split into train/test sets
3. **Train** three classifiers: Logistic Regression, Decision Tree, and SVM
4. **Evaluate and compare** model performance

## Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, built into scikit-learn
- **Samples:** 569
- **Features:** 30 numeric features computed from digitized images of breast-mass cell nuclei
- **Target:** Binary — malignant (`0`) or benign (`1`)

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|:--------:|:---------:|:------:|:--------:|
| **Logistic Regression** | **95.61%** | **0.957** | **0.956** | **0.956** |
| SVM | 94.74% | 0.952 | 0.947 | 0.947 |
| Decision Tree | 93.86% | 0.939 | 0.939 | 0.938 |

*Measured on an 80/20 train/test split with `random_state=42`.*

**Logistic Regression performed best** across every metric on this dataset.

## How to Run

```bash
# 1. (Optional) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python classification_models_comparison.py
```

## Tech Stack

- **Python 3**
- **scikit-learn** — datasets, models, and evaluation metrics
- **pandas** — data handling

## Possible Improvements

- Add feature scaling (`StandardScaler`) — likely to boost SVM and Logistic Regression
- Use cross-validation instead of a single split for more reliable metrics
- Add a confusion matrix and ROC curve for deeper evaluation
- Try ensemble models such as Random Forest or Gradient Boosting
