import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from xgboost import XGBClassifier
import os

data_path = "/Users/coopertammen/Desktop/CIS732/CIS-732-Term-Project/Spring 2026 Term Project/DataPrep/final_dataset.csv"
modeling_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(modeling_dir), "Results")

features = ["round", "delta_adjoe", "delta_adjde", "delta_sos", "seed_gap", "delta_win_pct_last10"]

df = pd.read_csv(data_path)
train_df = df[df["season"] <= 2022].reset_index(drop=True)
X_train = train_df[features].values
y_train = train_df["label"].values
groups = train_df["season"].values
cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups))

# Logistic Regression
print("Fitting Logistic Regression...")
pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
gs_lr = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]}, cv=cv_splits, scoring="neg_log_loss", n_jobs=-1, verbose=0)
gs_lr.fit(X_train, y_train)
lr_coefs = gs_lr.best_estimator_.named_steps["lr"].coef_[0]
print(f"  Best C: {gs_lr.best_params_['lr__C']}")

# Random Forest
print("Fitting Random Forest...")
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [100, 300, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
    cv=cv_splits, scoring="neg_log_loss", n_jobs=-1, verbose=0)
gs_rf.fit(X_train, y_train)
rf_imps = gs_rf.best_estimator_.feature_importances_
print(f"  Best params: {gs_rf.best_params_}")

# XGBoost
print("Fitting XGBoost...")
gs_xgb = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
    {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1, 0.3], "subsample": [0.8, 1.0]},
    cv=cv_splits, scoring="neg_log_loss", n_jobs=-1, verbose=0)
gs_xgb.fit(X_train, y_train)
xgb_imps = gs_xgb.best_estimator_.feature_importances_
print(f"  Best params: {gs_xgb.best_params_}")

# Normalize LR coefficients to [0,1] for avg_importance comparability
lr_abs = np.abs(lr_coefs)
lr_norm = lr_abs / lr_abs.sum()

avg_importance = (lr_norm + rf_imps + xgb_imps) / 3

out = pd.DataFrame({
    "feature": features,
    "lr_coef": lr_coefs.round(4),
    "rf_importance": rf_imps.round(4),
    "xgb_importance": xgb_imps.round(4),
    "avg_importance": avg_importance.round(4),
}).sort_values("avg_importance", ascending=False).reset_index(drop=True)

print("\nFeature Importance Summary:")
print(out.to_string(index=False))

out.to_csv(os.path.join(results_dir, "feature_importances.csv"), index=False)
print("\nSaved Results/feature_importances.csv")
