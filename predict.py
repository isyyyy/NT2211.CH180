import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load test data
data = pd.read_csv("data/test.csv")

# Normalize the data (assuming MinMaxScaler was used during training)
print("\n### Normalizing Data ###")
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

model = joblib.load("models/xgboost_model.pkl")
predictions = model.predict(data_normalized)

encoder = joblib.load('encoder.joblib')
predictions = encoder.inverse_transform(predictions)

print("\n### Saving Predictions ###")
# New dataset with predictions
data['Predictions'] = predictions
data.to_csv("results/predictions.csv", index=False)
