import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "model/house_price_model.pkl"

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

X = data[["area", "bedrooms", "age"]].copy()
y = data["price"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

FIT_INTERCEPT = True

model = LinearRegression(fit_intercept=FIT_INTERCEPT)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")