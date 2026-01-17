import streamlit as st
import pandas as pd
import joblib
import json

st.title("🏠 House Price Prediction Using ML")

# Load model and columns
model = joblib.load("house_price_model.pkl")
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

# User input
area = st.number_input("Area (sq ft)", 500, 5000, 1200)
bhk = st.number_input("BHK (Bedrooms)", 1, 10, 3)
bath = st.number_input("Bathrooms", 1, 10, 2)
location = st.selectbox("Location", ["Downtown", "Suburb", "Oldtown"])

# Create input dataframe
input_dict = {"area": area, "bhk": bhk, "bath": bath}
for loc in model_columns:
    if loc.startswith("location_"):
        input_dict[loc] = 1 if loc == f"location_{location}" else 0

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: {prediction:.2f}")
