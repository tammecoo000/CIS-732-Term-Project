import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix
import os

data_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project/DataPrep/final_dataset.csv"
out_dir = os.path.dirname(os.path.abspath(__file__))
feat_cols = ["round", "delta_adjoe", "delta_adjde", "delta_sos", "seed_gap", "delta_win_pct_last10"]

df = pd.read_csv(data_path)
train_df = df[df["season"] <= 2022].reset_index(drop=True)
test_df = df[df["season"] >= 2023].reset_index(drop=True)

X_train, y_train = train_df[feat_cols].values, train_df["label"].values
X_test,  y_test = test_df[feat_cols].values,  test_df["label"].values
groups = train_df["season"].values

gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train, y_train, groups=groups))

pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
param_grid = {"lr__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]}
gs = GridSearchCV(pipe, param_grid, cv=cv_splits, scoring="neg_log_loss", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print(f"\nBest C: {gs.best_params_['lr__C']}  |  CV log-loss: {-gs.best_score_:.4f}")

y_pred = gs.predict(X_test)
y_prob = gs.predict_proba(X_test)[:, 1]
print("\n=== Logistic Regression (CV-tuned) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss : {log_loss(y_test, y_prob):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
best_lr = gs.best_estimator_.named_steps["lr"]
print("\nCoefficients (standardized):")
for col, coef in zip(feat_cols, best_lr.coef_[0]):
    print(f"  {col:35s} {coef:+.4f}")

preds = test_df[["season", "round"]].copy()
preds["y_true"] = y_test
preds["y_pred"] = y_pred
preds["y_prob"] = y_prob
preds.to_csv(os.path.join(out_dir, "lr_preds.csv"), index=False)
print("\nSaved lr_preds.csv")
