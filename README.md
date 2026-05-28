# Classification Models Comparison

Train and compare three classic classification algorithms on the same dataset,
using consistent metrics — because choosing a model should be a measured
decision, not a guess.

## What it does

Trains and evaluates three models on the **Breast Cancer Wisconsin** dataset
(binary classification: malignant vs. benign):

- **Logistic Regression**
- **Decision Tree**
- **Support Vector Machine (SVM)**

Each model is scored on the same train/test split using four metrics:
**Accuracy, Precision, Recall, and F1-score.**

## Why these metrics

For a medical dataset, accuracy alone is misleading. Precision and recall expose
the cost of false positives vs. false negatives, and F1 balances the two — which
is exactly the trade-off that matters when picking a model for real use.

## Installation

```bash
pip install scikit-learn pandas matplotlib seaborn
```

## Usage

```bash
python "4 Implementing and evaluating classification models.py"
```

Prints each model's accuracy and the detailed precision/recall/F1 breakdown.

## Example output

```
Logistic Regression Accuracy: 95.61%
Decision Tree Accuracy: 93.86%
SVM Accuracy: 94.74%

Logistic Regression - Precision: 0.96, Recall: 0.96, F1 Score: 0.96
```

## Tech stack
Python · scikit-learn · pandas
