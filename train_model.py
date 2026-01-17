import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("house_prices.csv")

# One-hot encode location
X = df.drop("price", axis=1)
y = df["price"]

X = pd.get_dummies(X, columns=["location"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Evaluate
lr_mae = mean_absolute_error(y_test, lr_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print("Linear Regression MAE:", lr_mae)
print("Random Forest MAE:", rf_mae)

# Choose best model
best_model = rf if rf_mae < lr_mae else lr
joblib.dump(best_model, "house_price_model.pkl")
print("Best model saved as house_price_model.pkl")

# Save columns for Streamlit
import json
model_columns = list(X.columns)
with open("model_columns.json", "w") as f:
    f.write(json.dumps(model_columns))





















