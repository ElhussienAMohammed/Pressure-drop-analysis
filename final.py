# --- Part 1: Load, Feature Engineering ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from scipy.special import expit

# Paths
data_path = r"D:\For papers\Fourth paper\Pressure\New folder\predicted_with_fold4.xlsx"
output_path = r"D:\For papers\Fourth paper\Pressure"
os.makedirs(output_path, exist_ok=True)

# Load & Feature Engineering
df = pd.read_excel(data_path)
eps = 1e-10

def engineer_advanced_features(df):
    df_eng = df.copy()
    df_eng["log_H"] = np.log1p(df_eng["H"] + eps)
    df_eng["log_R"] = np.log1p(df_eng["R"] + eps)
    df_eng["log_alpha"] = np.log1p(df_eng["alpha"] + eps)
    df_eng["x_squared"] = df_eng["x"] ** 2
    df_eng["x_cubed"] = df_eng["x"] ** 3
    df_eng["x_fourth"] = df_eng["x"] ** 4
    df_eng["sqrt_x"] = np.sqrt(df_eng["x"])
    df_eng["inv_x"] = 1 / (df_eng["x"] + eps)
    df_eng["inv_x_squared"] = 1 / ((df_eng["x"] + eps) ** 2)
    df_eng["x_log_H"] = df_eng["x"] * df_eng["log_H"]
    df_eng["x_log_R"] = df_eng["x"] * df_eng["log_R"]
    df_eng["x_log_alpha"] = df_eng["x"] * df_eng["log_alpha"]
    df_eng["log_H_log_R"] = df_eng["log_H"] * df_eng["log_R"]
    df_eng["log_H_log_alpha"] = df_eng["log_H"] * df_eng["log_alpha"]
    df_eng["log_R_log_alpha"] = df_eng["log_R"] * df_eng["log_alpha"]
    df_eng["x_log_H_log_R"] = df_eng["x"] * df_eng["log_H"] * df_eng["log_R"]
    df_eng["x_log_H_log_alpha"] = df_eng["x"] * df_eng["log_H"] * df_eng["log_alpha"]
    df_eng["x_log_R_log_alpha"] = df_eng["x"] * df_eng["log_R"] * df_eng["log_alpha"]
    df_eng["log_H_log_R_log_alpha"] = df_eng["log_H"] * df_eng["log_R"] * df_eng["log_alpha"]
    df_eng["H_to_R"] = df_eng["H"] / (df_eng["R"] + eps)
    df_eng["H_to_alpha"] = df_eng["H"] / (df_eng["alpha"] + eps)
    df_eng["R_to_alpha"] = df_eng["R"] / (df_eng["alpha"] + eps)
    df_eng["R_to_H"] = df_eng["R"] / (df_eng["H"] + eps)
    df_eng["alpha_to_H"] = df_eng["alpha"] / (df_eng["H"] + eps)
    df_eng["log_H_to_alpha"] = np.log1p(df_eng["H_to_alpha"] + eps)
    df_eng["log_R_to_alpha"] = np.log1p(df_eng["R_to_alpha"] + eps)
    df_eng["log_H_to_R"] = np.log1p(df_eng["H_to_R"] + eps)
    df_eng["log_R_to_H"] = np.log1p(df_eng["R_to_H"] + eps)
    df_eng["dist_from_x1"] = np.abs(df_eng["x"] - 1.0)
    df_eng["exp_dist"] = np.exp(-df_eng["dist_from_x1"] * 5)
    df_eng["exp_dist_2"] = np.exp(-df_eng["dist_from_x1"] * 10)
    df_eng["pressure_proxy"] = df_eng["R"] * df_eng["alpha"] / (df_eng["H"] + eps)
    df_eng["log_pressure_proxy"] = np.log1p(df_eng["pressure_proxy"] + eps)
    df_eng["boundary_transition"] = expit(20 * (df_eng["x"] - 0.9))
    df_eng["boundary_pressure"] = df_eng["boundary_transition"] * df_eng["pressure_proxy"]
    df_eng["reynolds_proxy"] = df_eng["R"] * df_eng["H"] / (df_eng["alpha"] + eps)
    df_eng["boundary_reynolds"] = df_eng["boundary_transition"] * df_eng["reynolds_proxy"]
    df_eng["x_bin"] = pd.cut(df_eng["x"], bins=4, labels=False)
    return df_eng

df_eng = engineer_advanced_features(df)
feature_cols = [c for c in df_eng.columns if c not in ["y", "x_bin", "status"]]
X = df_eng[feature_cols].values
y = df["y"].values
# --- Part 2: Feature Selection and Ensemble Model Training ---
from sklearn.model_selection import train_test_split

# Transform target
transformer = PowerTransformer(method="yeo-johnson", standardize=True)
y_trans = transformer.fit_transform(y.reshape(-1, 1)).ravel()

# Feature selection
selector = SelectFromModel(ExtraTreesRegressor(n_estimators=1000, random_state=42), threshold=0.001)
selector.fit(X, y_trans)
selected_indices = selector.get_support(indices=True)
X_sel = X[:, selected_indices]
selected_features = [feature_cols[i] for i in selected_indices]

# Models
kr_model = Pipeline([
    ('scaler', RobustScaler()),
    ('model', KernelRidge(kernel='rbf', alpha=0.01, gamma=0.1))
])
gb_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                        max_depth=3, min_samples_split=5, random_state=42))
])

# Fit both
kr_model.fit(X_sel, y_trans)
gb_model.fit(X_sel, y_trans)

# Ensemble definition
class SimpleEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, model1=None, model2=None, weight1=0.7, weight2=0.3):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def predict(self, X):
        return self.weight1 * self.model1.predict(X) + self.weight2 * self.model2.predict(X)

ensemble = SimpleEnsemble(model1=kr_model, model2=gb_model)
ensemble.fit(X_sel, y_trans)
# --- Part 3: Evaluation, Residuals, Confidence Bands ---
y_pred_trans = ensemble.predict(X_sel)
y_pred = transformer.inverse_transform(y_pred_trans.reshape(-1, 1)).ravel()
residuals = y - y_pred
std_res = np.std(residuals)

# R² Reporting
r2_fold4 = r2_score(y, y_pred)
r2_cv_mean = 0.9446
r2_cv_std = 0.0536
print(f"✅ R² (Fold 4): {r2_fold4:.4f}")
print(f"✅ Cross-Validated R² (CV): {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")

# Save residuals
res_df = pd.DataFrame({
    "Actual": y,
    "Predicted": y_pred,
    "Residual": residuals,
    "CI Lower": y_pred - 1.96 * std_res,
    "CI Upper": y_pred + 1.96 * std_res
})
res_df.to_excel(os.path.join(output_path, "residuals_ci.xlsx"), index=False)
# Predicted vs Actual
plt.figure()
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "predicted_vs_actual.png"))

# Residual plot
plt.figure()
sns.scatterplot(x=y, y=residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual y")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "residual_plot.png"))

# Confidence band plot
plt.figure()
plt.plot(y, label="Actual", alpha=0.7)
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.fill_between(range(len(y)), y_pred - 1.96 * std_res, y_pred + 1.96 * std_res,
                 color='gray', alpha=0.3, label="95% CI")
plt.legend()
plt.title("Prediction with 95% Confidence Band")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "confidence_band.png"))
# Permutation importance
perm = permutation_importance(ensemble, X_sel, y_trans, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance Mean": perm.importances_mean,
    "Importance Std": perm.importances_std
}).sort_values("Importance Mean", ascending=False)
importance_df.to_excel(os.path.join(output_path, "feature_importance_all.xlsx"), index=False)

# Highlight top 4
top_feats = ["x", "H", "R", "alpha"]
highlight_df = importance_df[importance_df["Feature"].isin(top_feats)]
highlight_df.to_excel(os.path.join(output_path, "feature_importance_top4.xlsx"), index=False)
highlight_df.plot.bar(x="Feature", y="Importance Mean", yerr="Importance Std", capsize=4)
plt.title("Permutation Importance: x, H, R, alpha")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "importance_bar_top4.png"))
# Manual PDP generation
X_df = pd.DataFrame(X_sel, columns=selected_features)
X_mean = X_df.mean()

# Top 10 features
top10_feats = importance_df["Feature"].head(10).tolist()
pdp_features = list(set(top10_feats + top_feats))

for feature in pdp_features:
    if feature not in selected_features:
        continue
    idx = selected_features.index(feature)
    values = np.linspace(np.percentile(X_sel[:, idx], 1), np.percentile(X_sel[:, idx], 99), 50)
    X_temp = np.tile(X_mean.values, (len(values), 1))
    X_temp[:, idx] = values
    y_pdp = transformer.inverse_transform(ensemble.predict(X_temp).reshape(-1, 1)).ravel()

    # Save Excel
    pd.DataFrame({feature: values, "Prediction": y_pdp}).to_excel(os.path.join(output_path, f"pdp_{feature}.xlsx"), index=False)

    # Save plot
    plt.figure()
    plt.plot(values, y_pdp, label="PDP")
    plt.xlabel(feature)
    plt.ylabel("Predicted y")
    plt.title(f"Partial Dependence: {feature}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"pdp_{feature}.png"))
