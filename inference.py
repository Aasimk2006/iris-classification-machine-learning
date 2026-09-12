import joblib
import pandas as pd

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
model = joblib.load("models/iris_best_model.joblib")

new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=features)
print("Predicted species:", model.predict(new_flower)[0])
