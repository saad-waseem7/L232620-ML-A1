import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "model/house_price_model.pkl"

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

X = data[["area", "bedrooms", "age"]]
y = data["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")