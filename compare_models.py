#Step 1: Set up the environment
#pip install scikit-learn
#pip install pandas
#pip install matplotlib seaborn

#Step 2: Load and explore the dataset
import pandas as pd

# Load the dataset
# Load Breast Cancer dataset and convert to DataFrame
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Explore the dataset
print(df.head())
print(df.info())

#Step 3: Preprocess the data
from sklearn.model_selection import train_test_split

# Handle missing values (example: filling missing values with the median)
#df.fillna(df.median(), inplace=True)

# Split the data into features and labels
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train and compare models with proper scaling
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
}

results = []
for name, model in models.items():
    # Scaling lives inside the pipeline, so it's fit on train data only.
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
    })

# Step 5: Compare model performance
comparison = pd.DataFrame(results).set_index("Model").round(4)
print(comparison)
print(f"\nBest model by F1: {comparison['F1'].idxmax()}")
