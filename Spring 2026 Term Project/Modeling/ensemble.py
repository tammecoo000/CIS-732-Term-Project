import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import os

modeling_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(modeling_dir), "Results")
os.makedirs(results_dir, exist_ok=True)

lr  = pd.read_csv(os.path.join(modeling_dir, "lr_preds.csv"))
rf  = pd.read_csv(os.path.join(modeling_dir, "rf_preds.csv"))
xgb = pd.read_csv(os.path.join(modeling_dir, "xgb_preds.csv"))

assert (lr["y_true"].values == rf["y_true"].values).all(), "y_true mismatch LR vs RF"
assert (lr["y_true"].values == xgb["y_true"].values).all(), "y_true mismatch LR vs XGB"

ens = lr[["season", "round", "y_true"]].copy()
ens["y_prob"] = (lr["y_prob"] + rf["y_prob"] + xgb["y_prob"]) / 3
ens["y_pred"] = (ens["y_prob"] >= 0.5).astype(int)

acc = accuracy_score(ens["y_true"], ens["y_pred"])
ll  = log_loss(ens["y_true"], ens["y_prob"])
auc = roc_auc_score(ens["y_true"], ens["y_prob"])

print(f"Ensemble  Accuracy: {acc:.4f}  Log Loss: {ll:.4f}  AUC: {auc:.4f}")

ens.to_csv(os.path.join(results_dir, "ensemble_preds.csv"), index=False)

metrics = pd.DataFrame([{"model": "Ensemble", "accuracy": acc, "log_loss": ll, "auc": auc}])
metrics.to_csv(os.path.join(results_dir, "ensemble_metrics.csv"), index=False)

print("Saved Results/ensemble_preds.csv and Results/ensemble_metrics.csv")
