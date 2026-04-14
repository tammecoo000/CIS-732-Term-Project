import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix
import os

data_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project/DataPrep/final_dataset.csv"
out_dir = os.path.dirname(os.path.abspath(__file__))
features = ["round", "delta_adjoe", "delta_adjde", "delta_sos", "seed_gap", "delta_win_pct_last10"]

df = pd.read_csv(data_path)
train_df = df[df["season"] <= 2022].reset_index(drop=True)
test_df = df[df["season"] >= 2023].reset_index(drop=True)

X_train = train_df[features].values
y_train = train_df["label"].values
X_test = test_df[features].values
y_test = test_df["label"].values
seasons = train_df["season"].values

print(f"Train: {len(train_df)} games | Test: {len(test_df)} games")

cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=seasons))

param_grid = {
    "n_estimators": [100, 300],
    "max_depth":    [3, 5],
    "learning_rate":[0.05, 0.1, 0.3],
    "subsample":    [0.8, 1.0],
}
gs = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
    param_grid,
    cv=cv_splits,
    scoring="neg_log_loss",
    n_jobs=-1,
    verbose=1,
)
gs.fit(X_train, y_train)

print(f"\nBest params: {gs.best_params_}  |  CV log-loss: {-gs.best_score_:.4f}")

y_pred = gs.predict(X_test)
y_prob = gs.predict_proba(X_test)[:, 1]

print("\n=== XGBoost (CV-tuned) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss : {log_loss(y_test, y_prob):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))

print("\nFeature importances:")
importances = zip(features, gs.best_estimator_.feature_importances_)
for feat, imp in sorted(importances, key=lambda x: -x[1]):
    print(f"  {feat:35s} {imp:.4f}")

preds = test_df[["season", "round"]].copy()
preds["y_true"] = y_test
preds["y_pred"] = y_pred
preds["y_prob"] = y_prob
preds.to_csv(os.path.join(out_dir, "xgb_preds.csv"), index=False)
print("\nSaved xgb_preds.csv")
