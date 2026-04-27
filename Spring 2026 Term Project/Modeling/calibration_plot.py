import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os

modeling_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(modeling_dir), "Results")

models = {
    "Logistic Regression": pd.read_csv(os.path.join(modeling_dir, "lr_preds.csv")),
    "Random Forest": pd.read_csv(os.path.join(modeling_dir, "rf_preds.csv")),
    "XGBoost": pd.read_csv(os.path.join(modeling_dir, "xgb_preds.csv")),
    "Ensemble": pd.read_csv(os.path.join(results_dir,  "ensemble_preds.csv")),
}

colors  = ["steelblue", "forestgreen", "firebrick", "darkorange"]
markers = ["o", "s", "^", "D"]

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

for (name, df), color, marker in zip(models.items(), colors, markers):
    frac_pos, mean_pred = calibration_curve(df["y_true"], df["y_prob"], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker=marker, color=color, linewidth=1.8, markersize=6, label=name)

ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives (Actual Win Rate)", fontsize=12)
ax.set_title("Probability Calibration — March Madness Win Prediction", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.tick_params(labelsize=10)
plt.tight_layout()

out_path = os.path.join(results_dir, "fig_calibration.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
