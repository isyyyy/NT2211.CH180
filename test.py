import os
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load test data
print("\n### Loading Testing Data ###")
X_test = pd.read_csv("./data/X_test.csv")
y_test = pd.read_csv("./data/y_test.csv").squeeze()

# Normalize the test data
print("\n### Normalizing Testing Data ###")
scaler = MinMaxScaler()
X_test_normalized = scaler.fit_transform(X_test)

# List all saved models in the 'models' directory
models_dir = "models"
saved_models = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith("_model.pkl")]

# Results storage
testing_results = []

# Test each model
print("\n### Testing All Models ###")
for model_file in saved_models:
    model_name = os.path.basename(model_file).replace("_model.pkl", "").replace("_", " ").title()
    print(f"\nLoading and Testing Model: {model_name}")
    model = joblib.load(model_file)
    y_pred = model.predict(X_test_normalized)

    # Evaluate model
    acc = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    precision = classification_rep["weighted avg"]["precision"]
    recall = classification_rep["weighted avg"]["recall"]
    f1 = classification_rep["weighted avg"]["f1-score"]

    # Store the results
    testing_results.append({
        "Model": model_name,
        "Accuracy": round(acc, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-Score": round(f1, 2)
    })

    # Print evaluation results
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=2))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save testing results to CSV
results_df = pd.DataFrame(testing_results)
results_df.to_csv("results/testing_results.csv", index=False)
print("\nTesting results have been saved to 'testing_results.csv'")