import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

round_names = {
    0: "Play-In",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
}

models = {
    "Logistic Regression": "lr_preds.csv",
    "Random Forest":       "rf_preds.csv",
    "XGBoost":             "xgb_preds.csv",
}

# Load all prediction files
preds = {
    name: pd.read_csv(os.path.join(out_dir, fname))
    for name, fname in models.items()
}

# Overall comparison table
print("=" * 62)
print(f"{'Model':<25} {'Accuracy':>10} {'Log Loss':>10} {'AUC':>10}")
print("-" * 62)
for name, df in preds.items():
    acc = accuracy_score(df["y_true"], df["y_pred"])
    ll  = log_loss(df["y_true"], df["y_prob"])
    auc = roc_auc_score(df["y_true"], df["y_prob"])
    print(f"{name:<25} {acc:>10.4f} {ll:>10.4f} {auc:>10.4f}")
print("=" * 62)

# Round-by-round accuracy
print("\nRound-by-Round Accuracy")
print("-" * 72)
header = f"{'Round':<15}" + "".join(f"{m:>19}" for m in models)
print(header)
print("-" * 72)

all_rounds = sorted(next(iter(preds.values()))["round"].unique())
for r in all_rounds:
    label = round_names.get(r, f"Round {r}")
    row = f"{label:<15}"
    for df in preds.values():
        sub = df[df["round"] == r]
        acc = accuracy_score(sub["y_true"], sub["y_pred"])
        row += f"{acc:.4f} ({len(sub):3d}g)  "
    print(row)
print("-" * 72)

# Season-by-season accuracy
print("\nSeason-by-Season Accuracy (Test Set)")
print("-" * 72)
for season in sorted(next(iter(preds.values()))["season"].unique()):
    row = f"{int(season):<15}"
    for df in preds.values():
        sub = df[df["season"] == season]
        acc = accuracy_score(sub["y_true"], sub["y_pred"])
        row += f"{acc:.4f} ({len(sub):3d}g)  "
    print(row)
print("-" * 72)
