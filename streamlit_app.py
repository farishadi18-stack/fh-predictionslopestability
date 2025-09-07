# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from smogn import smoter

# ----------------------------
# Title
# ----------------------------
st.title("Preliminary Slope Stability Assessment Tool")
st.write("Estimate Factor of Safety (FoS) using soil and nail parameters.")

# ----------------------------
# User Input Section
# ----------------------------
c = st.number_input("Cohesion (kPa)", min_value=0.0, step=0.5)
phi = st.number_input("Friction Angle (°)", min_value=0.0, max_value=60.0, step=0.5)
gamma = st.number_input("Unit Weight (kN/m³)", min_value=10.0, max_value=25.0, step=0.1)

nail_length = st.number_input("Nail Length (m)", min_value=3.0, step=1.0)
nail_diameter = st.number_input("Drillhole Diameter (mm)", min_value=50.0, step=5.0)
nail_inclination = st.number_input("Inclination (°)", min_value=0.0, max_value=30.0, step=1.0)

slope_angle = st.number_input("Slope Angle (°)", min_value=15.0, max_value=90.0, step=1.0)

# ----------------------------
# Placeholder Data
# ----------------------------
# (Replace with your real dataset in CSV)
try:
    df = pd.read_csv("new treated slope.xlsx")
except:
    st.warning("No dataset found. Please upload soil_nail_data.csv to GitHub repo.")
    df = pd.DataFrame(columns=[
        "c","phi","gamma","nail_length","nail_diameter","nail_inclination","slope_angle","FoS"
    ])

# ----------------------------
# Preprocessing: SMOGN
# ----------------------------
if not df.empty:
    df_balanced = smoter(
        data=df,
        y="FoS"  # Target column
    )

    X = df_balanced.drop("FoS", axis=1)
    y = df_balanced["FoS"]

    # ----------------------------
    # Random Forest + KFold = 10
    # ----------------------------
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    st.write(f"Model CV R² Score: {scores.mean():.3f}")

    # Fit final model
    model.fit(X, y)

    # ----------------------------
    # Prediction
    # ----------------------------
    input_data = np.array([[c, phi, gamma, nail_length, nail_diameter, nail_inclination, slope_angle]])
    fos_pred = model.predict(input_data)[0]
    st.success(f"Predicted Factor of Safety (FoS): {fos_pred:.3f}")
