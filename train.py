import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, f1_score, recall_score,
                             matthews_corrcoef, precision_score)
from xgboost import XGBClassifier

# Section 1: Load Data
print("### Loading Data ###")
X_train = pd.read_csv("./data/X_train.csv")
y_train = pd.read_csv("./data/y_train.csv").squeeze()  # Convert to Series
X_test = pd.read_csv("./data/X_test.csv")
y_test = pd.read_csv("./data/y_test.csv").squeeze()  # Convert to Series

# Create the 'models' directory if it doesn't exist
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# List of models to train and evaluate
models = {
    # "Random Forest": RandomForestClassifier(random_state=41, n_estimators=100, max_depth=20),
    # "Logistic Regression": LogisticRegression(max_iter=1000, random_state=41),
    # "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
}

# Section 2: Evaluate and Save All Models
print("\n### Training and Saving All Models ###")
evaluation_results = []

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the trained model
    model_filename = os.path.join(models_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(model, model_filename)
    print(f"Model '{model_name}' has been saved as '{model_filename}'")

    # Metrics Calculation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # TPR (True Positive Rate) and FPR (False Positive Rate) for each class
    tpr_list = []
    fpr_list = []
    for idx, cls in enumerate(sorted(set(y_test))):
        tp = cm[idx, idx]  # True Positives
        fn = cm[idx, :].sum() - tp  # False Negatives
        fp = cm[:, idx].sum() - tp  # False Positives
        tn = cm.sum() - (tp + fn + fp)  # True Negatives

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Append Results
    evaluation_results.append({
        "Model": model_name,
        "Accuracy": round(acc, 2),
        "F1-Score": round(f1, 2),
        "MCC": round(mcc, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "TPR": round(np.mean(tpr_list), 2),  # Mean TPR across all classes
        "FPR": round(np.mean(fpr_list), 2)   # Mean FPR across all classes
    })

    # Confusion Matrix and Classification Report
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    print(f"\nClassification Report for {model_name}:\n{classification_report(y_test, y_pred)}")

# Section 3: Save Evaluation Results
results_df = pd.DataFrame(evaluation_results)
print("\n### Evaluation Metrics for All Models ###")
print(results_df)

# Format results for consistent decimal places in CSV
results_df = results_df.round(2)
results_df.to_csv("results/model_evaluation_results.csv", index=False)
print("Evaluation metrics have been saved to 'model_evaluation_results.csv'")