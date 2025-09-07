# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# Title
# ----------------------------
st.title("Preliminary Slope Stability Assessment Tool")
st.write("Estimate Factor of Safety (FoS) using soil shear strength and nail parameters.")

# ----------------------------
# User Input
# ----------------------------
c = st.number_input("Cohesion (kPa)", min_value=0.0, max_value=60.0, step=0.5)
phi = st.number_input("Friction Angle (°)", min_value=0.0, max_value=60.0, step=0.5)
nail_length = st.number_input("Nail Length (m)", min_value=3.0, max_value=30.0, step=1.0)
nail_diameter = st.number_input("Drillhole Diameter (mm)", min_value=50.0, max_value=200.0, step=5.0)
nail_inclination = st.number_input("Inclination (°)", min_value=0.0, max_value=30.0, step=1.0)
slope_angle = st.number_input("Slope Angle (°)", min_value=15.0, max_value=90.0, step=1.0)

# ----------------------------
# Load Dataset
# ----------------------------
try:
    df = pd.read_csv("new treated slope.xlsx")
except:
    st.error("❌ Dataset not found. Please upload 'new treated slope.slsx' to your repo.")
    st.stop()

# ----------------------------
# Train Final Model
# ----------------------------
X = df[["Cohesion", "Friction_Angle", "Nail_Length",
        "Drillhole_Diameter", "Nail_Inclination", "Slope_Angle"]]
y = df["Factor_of_Safety"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict FoS"):
    input_data = np.array([[c, phi, nail_length, nail_diameter, nail_inclination, slope_angle]])
    fos_pred = model.predict(input_data)[0]
    st.success(f"Predicted Factor of Safety (FoS): {fos_pred:.3f}")
