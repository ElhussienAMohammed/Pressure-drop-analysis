import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fold4_ensemble_model.pkl")
transformer = joblib.load("target_transformer.pkl")
selected_features = joblib.load("selected_features.pkl")

# Constants
AIR_DENSITY = 1.225
D_H = 66.7  # mm (fixed)
eps = 1e-10

# Title
st.title("🔬 Cyclone Pressure Drop Estimator")
st.markdown("Enter your particle properties to estimate cyclone pressure drop for spherical particles.")

# Inputs
phi = st.slider("1️⃣ Particle Shape Factor (Φ)", 0.5, 1.0, 0.85, step=0.01)
rho_s = st.slider("2️⃣ Particle Density ρₛ (kg/m³)", 700, 3320, 1500)
d_p = st.slider("3️⃣ Particle Diameter dₚ (µm)", 0.1, 10.0, 1.0)
alpha_p = st.slider("4️⃣ Volume Fraction αₚ", 1e-5, 1e-3, 1e-4, format="%.6f")

# Computed variables
X_input = phi
R = rho_s / AIR_DENSITY
H = d_p / (D_H * 1000)
alpha = alpha_p

# Create feature-engineered input
def engineer_features(X, H, R, alpha):
    df = pd.DataFrame({
        "x": [X], "H": [H], "R": [R], "alpha": [alpha]
    })
    df["log_H"] = np.log1p(df["H"] + eps)
    df["log_R"] = np.log1p(df["R"] + eps)
    df["log_alpha"] = np.log1p(df["alpha"] + eps)
    df["x_squared"] = df["x"] ** 2
    df["x_cubed"] = df["x"] ** 3
    df["x_fourth"] = df["x"] ** 4
    df["sqrt_x"] = np.sqrt(df["x"])
    df["inv_x"] = 1 / (df["x"] + eps)
    df["inv_x_squared"] = 1 / ((df["x"] + eps) ** 2)
    df["x_log_H"] = df["x"] * df["log_H"]
    df["x_log_R"] = df["x"] * df["log_R"]
    df["x_log_alpha"] = df["x"] * df["log_alpha"]
    df["log_H_log_R"] = df["log_H"] * df["log_R"]
    df["log_H_log_alpha"] = df["log_H"] * df["log_alpha"]
    df["log_R_log_alpha"] = df["log_R"] * df["log_alpha"]
    df["x_log_H_log_R"] = df["x"] * df["log_H"] * df["log_R"]
    df["x_log_H_log_alpha"] = df["x"] * df["log_H"] * df["log_alpha"]
    df["x_log_R_log_alpha"] = df["x"] * df["log_R"] * df["log_alpha"]
    df["log_H_log_R_log_alpha"] = df["log_H"] * df["log_R"] * df["log_alpha"]
    df["H_to_R"] = df["H"] / (df["R"] + eps)
    df["H_to_alpha"] = df["H"] / (df["alpha"] + eps)
    df["R_to_alpha"] = df["R"] / (df["alpha"] + eps)
    df["R_to_H"] = df["R"] / (df["H"] + eps)
    df["alpha_to_H"] = df["alpha"] / (df["H"] + eps)
    df["log_H_to_alpha"] = np.log1p(df["H_to_alpha"] + eps)
    df["log_R_to_alpha"] = np.log1p(df["R_to_alpha"] + eps)
    df["log_H_to_R"] = np.log1p(df["H_to_R"] + eps)
    df["log_R_to_H"] = np.log1p(df["R_to_H"] + eps)
    df["dist_from_x1"] = np.abs(df["x"] - 1.0)
    df["exp_dist"] = np.exp(-df["dist_from_x1"] * 5)
    df["pressure_proxy"] = df["R"] * df["alpha"] / (df["H"] + eps)
    df["log_pressure_proxy"] = np.log1p(df["pressure_proxy"] + eps)
    df["boundary_transition"] = 1 / (1 + np.exp(-20 * (df["x"] - 0.9)))
    df["boundary_pressure"] = df["boundary_transition"] * df["pressure_proxy"]
    df["reynolds_proxy"] = df["R"] * df["H"] / (df["alpha"] + eps)
    df["boundary_reynolds"] = df["boundary_transition"] * df["reynolds_proxy"]
    return df[selected_features]

# Predict
input_data = engineer_features(X_input, H, R, alpha)
pred_trans = model.predict(input_data)
pred = transformer.inverse_transform(pred_trans.reshape(-1, 1)).ravel()[0]

# Display Result
st.markdown("## ✅ Your cyclone pressure drop efficiency:")
st.success(f"**y = {pred:.6f}**")
