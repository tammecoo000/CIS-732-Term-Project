import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import os

modeling_dir = os.path.dirname(os.path.abspath(__file__))
results_dir  = os.path.join(os.path.dirname(modeling_dir), "Results")

lr = pd.read_csv(os.path.join(modeling_dir, "lr_preds.csv"))
rf = pd.read_csv(os.path.join(modeling_dir, "rf_preds.csv"))
xgb = pd.read_csv(os.path.join(modeling_dir, "xgb_preds.csv"))
ens = pd.read_csv(os.path.join(results_dir,  "ensemble_preds.csv"))

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb,
    "Ensemble": ens,
}

for name, df in models.items():
    df["correct"] = (df["y_pred"] == df["y_true"]).astype(int)

pairs = [
    ("Logistic Regression", "Random Forest"),
    ("Logistic Regression", "XGBoost"),
    ("Random Forest", "XGBoost"),
    ("Logistic Regression", "Ensemble"),
    ("Random Forest", "Ensemble"),
    ("XGBoost", "Ensemble"),
]

rows = []
print(f"\n{'Pair':<40} {'b':>5} {'c':>5} {'chi2':>8} {'p-value':>10} {'Reject H0':>10}")
print("-" * 82)

for name_a, name_b in pairs:
    a = models[name_a]["correct"].values
    b = models[name_b]["correct"].values

    b_count = int(((a == 1) & (b == 0)).sum())  # A correct, B wrong
    c_count = int(((a == 0) & (b == 1)).sum())  # A wrong, B correct

    table = np.array([[((a==1)&(b==1)).sum(), b_count], [c_count, ((a==0)&(b==0)).sum()]])

    result  = mcnemar(table, exact=False, correction=True)
    reject  = result.pvalue < 0.05
    label   = f"{name_a} vs {name_b}"

    print(f"{label:<40} {b_count:>5} {c_count:>5} {result.statistic:>8.4f} {result.pvalue:>10.4f} {'YES' if reject else 'NO':>10}")
    rows.append({
        "model_a": name_a,
        "model_b": name_b,
        "b": b_count,
        "c": c_count,
        "chi2": round(result.statistic, 4),
        "p_value": round(result.pvalue, 4),
        "reject_h0":  reject,
    })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(results_dir, "mcnemar_results.csv"), index=False)
print("\nSaved Results/mcnemar_results.csv")
