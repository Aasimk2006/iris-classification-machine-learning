# Iris Classification

A complete machine-learning classification project using the classic Iris dataset.

## Goal
Predict Iris species from:
- sepal length
- sepal width
- petal length
- petal width

## Dataset
Requested Kaggle source:
https://www.kaggle.com/datasets/bhanupratapbiswas/iris-classification-dataset

The execution environment could not directly download the Kaggle archive, so the included `data/iris.csv` uses the canonical 150-row Iris dataset with the same four features and three species. Replace that CSV with the downloaded Kaggle CSV if required, then rerun the notebook.

## Models
- k-NN (k=5) with StandardScaler
- Logistic Regression with StandardScaler
- Decision Tree (max_depth=4)

## Evaluation
The notebook reports:
- accuracy
- weighted precision
- weighted recall
- confusion matrices
- classification reports

## Result on the included fixed split
All three models achieve approximately 93.33% accuracy on the 80/20 stratified split (`random_state=42`). The notebook automatically selects the best model using accuracy, then precision and recall as tie-breakers.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook iris_classification.ipynb
```

## Inference
```python
import joblib
import pandas as pd

model = joblib.load("models/iris_best_model.joblib")

new_flower = pd.DataFrame([[
    5.1, 3.5, 1.4, 0.2
]], columns=[
    "sepal_length", "sepal_width", "petal_length", "petal_width"
])

print("Predicted species:", model.predict(new_flower)[0])
```

Expected result for this example: `setosa`.

## Structure
```text
iris-classification-machine-learning/
├── data/
│   └── iris.csv
├── models/
│   └── iris_best_model.joblib
├── iris_classification.ipynb
├── inference.py
├── README.md
└── requirements.txt
```
