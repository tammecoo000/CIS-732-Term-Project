import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import os

modeling_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(modeling_dir), "Results")

lr = pd.read_csv(os.path.join(modeling_dir, "lr_preds.csv"))
rf = pd.read_csv(os.path.join(modeling_dir, "rf_preds.csv"))
xgb = pd.read_csv(os.path.join(modeling_dir, "xgb_preds.csv"))
ens = pd.read_csv(os.path.join(results_dir,  "ensemble_preds.csv"))

all_models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb,
    "Ensemble": ens,
}

mcnemar = pd.read_csv(os.path.join(results_dir, "mcnemar_results.csv"))
feat_imp = pd.read_csv(os.path.join(results_dir, "feature_importances.csv"))

round_names = {
    0: "Play-In", 1: "Round of 64", 2: "Round of 32",
    3: "Sweet 16", 4: "Elite Eight", 5: "Final Four", 6: "Championship",
}

# --- 5a: Overall comparison table ---
rows = []
for name, df in all_models.items():
    acc = accuracy_score(df["y_true"], df["y_pred"])
    ll  = log_loss(df["y_true"], df["y_prob"])
    auc = roc_auc_score(df["y_true"], df["y_prob"])
    match = mcnemar[(mcnemar["model_a"] == name) & (mcnemar["model_b"] == "Ensemble")]
    if match.empty:
        match = mcnemar[(mcnemar["model_a"] == "Ensemble") & (mcnemar["model_b"] == name)]
    p_val = round(match["p_value"].values[0], 4) if not match.empty else "—"
    rows.append({"Model": name, "Accuracy": round(acc, 4), "Log Loss": round(ll, 4), "AUC": round(auc, 4),
                "McNemar p-value (vs Ensemble)": p_val})

overall = pd.DataFrame(rows)
overall.to_csv(os.path.join(results_dir, "table_overall_comparison.csv"), index=False)
print("Overall Comparison:")
print(overall.to_string(index=False))

# --- 5b: Round-by-round accuracy ---
rounds = sorted(lr["round"].unique())
round_rows = []
for r in rounds:
    row = {"Round": round_names.get(r, f"Round {r}")}
    for name, df in all_models.items():
        sub = df[df["round"] == r]
        row[name] = round(accuracy_score(sub["y_true"], sub["y_pred"]), 4)
    row["n"] = len(lr[lr["round"] == r])
    round_rows.append(row)

round_df = pd.DataFrame(round_rows)
round_df.to_csv(os.path.join(results_dir, "table_round_accuracy.csv"), index=False)
print("\nRound-by-Round Accuracy:")
print(round_df.to_string(index=False))

# --- 5c: Season-by-season accuracy ---
seasons = sorted(lr["season"].unique())
season_rows = []
for s in seasons:
    row = {"Season": int(s)}
    for name, df in all_models.items():
        sub = df[df["season"] == s]
        row[name] = round(accuracy_score(sub["y_true"], sub["y_pred"]), 4)
    row["n"] = len(lr[lr["season"] == s])
    season_rows.append(row)

season_df = pd.DataFrame(season_rows)
season_df.to_csv(os.path.join(results_dir, "table_season_accuracy.csv"), index=False)
print("\nSeason-by-Season Accuracy:")
print(season_df.to_string(index=False))

# --- 5d: Feature importance table ---
feat_out = feat_imp.copy()
feat_out.insert(0, "Rank", range(1, len(feat_out) + 1))
feat_out.to_csv(os.path.join(results_dir, "table_feature_importance.csv"), index=False)
print("\nFeature Importance:")
print(feat_out.to_string(index=False))

# --- 5e: Verify calibration figure ---
cal_path = os.path.join(results_dir, "fig_calibration.png")
size_kb = os.path.getsize(cal_path) // 1024
print(f"\nCalibration figure exists: {cal_path} ({size_kb} KB)")

print("\nAll report outputs saved to Results/")
