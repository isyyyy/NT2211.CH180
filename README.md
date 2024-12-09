# Model Training and Testing
This project provides a framework for training multiple machine learning models, saving them, and evaluating their performance using a structured approach.

It supports saving all models and their evaluation metrics, as well as testing and comparing results.

## Install the required packages
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## EDA
Run notebook EDA.py to perform exploratory data analysis on the provided dataset 
Then it will save the processed data to the data/ directory.
- X_train.csv
- y_train.csv
- X_test.csv
- y_test.csv



## Train the models
Run train.py to train multiple machine learning models, evaluate their performance, and save them to the models/ directory.
```bash
python train.py
```

- Models used:
  - XGBoost
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors
- Key metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

The results are saved in model_evaluation_results.csv.


## Test the models
Run test.py to load the saved models from the models/ directory, test them on the provided test dataset (X_test.csv, y_test.csv), and evaluate their performance.
```bash
python test.py
```


## Predict with test data
Run predict.py to load the saved models from the models/ directory, predict the target variable for the provided test dataset (X_test.csv), and save the predictions to the predictions/ directory.
```bash
python predict.py
```

## Results
The best model is XGBoost with an accuracy of 0.9 against the test dataset.

