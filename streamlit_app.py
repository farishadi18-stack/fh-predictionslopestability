# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import smogn

# ----------------------------
# Title
# ----------------------------
st.title("Preliminary Slope Stability Assessment Tool")
st.write("Estimate Factor of Safety (FoS) using soil shear strength and nail parameters.")

# ----------------------------
# User Input
# ----------------------------
c = st.number_input("Cohesion (kPa)", min_value=0.0, step=0.5)
phi = st.number_input("Friction Angle (°)", min_value=0.0, max_value=60.0, step=0.5)
nail_length = st.number_input("Nail Length (m)", min_value=3.0, step=1.0)
nail_diameter = st.number_input("Drillhole Diameter (mm)", min_value=50.0, step=5.0)
nail_inclination = st.number_input("Inclination (°)", min_value=0.0, max_value=30.0, step=1.0)
slope_angle = st.number_input("Slope Angle (°)", min_value=15.0, max_value=90.0, step=1.0)

# ----------------------------
# Load Dataset
# ----------------------------
try:
    df = pd.read_csv("new treated slope.csv")
    st.success("✅ Dataset loaded successfully")
except:
    st.error("❌ Dataset not found. Please upload 'new treated slope.csv' to your repo.")
    st.stop()

# ----------------------------
# Balance with SMOGN
# ----------------------------
with st.spinner("Balancing dataset with SMOGN..."):
    df_balanced = smogn.smoter(data=df, y="Factor_of_Safety")

X = df_balanced[["Cohesion", "Friction_Angle", "Nail_Length", 
                 "Drillhole_Diameter", "Nail_Inclination", "Slope_Angle"]]
y = df_balanced["Factor_of_Safety"]

# ----------------------------
# Train Model with K-Fold CV
# ----------------------------
with st.spinner("Training Random Forest model..."):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    model.fit(X, y)

st.info(f"Model 10-Fold CV R² Score: {scores.mean():.3f}")

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict FoS"):
    input_data = np.array([[c, phi, nail_length, nail_diameter, nail_inclination, slope_angle]])
    fos_pred = model.predict(input_data)[0]
    st.success(f"Predicted Factor of Safety (FoS): {fos_pred:.3f}")

