import pandas as pd

data_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project/DataPrep/final_dataset.csv"
df = pd.read_csv(data_path)

feature_cols = ["round", "delta_adjoe", "delta_adjde", "delta_sos", "seed_gap", "delta_win_pct_last10"]

train = df[df["season"] <= 2022].reset_index(drop=True)
test  = df[df["season"] >= 2023].reset_index(drop=True)

X_train, y_train = train[feature_cols], train["label"]
X_test,  y_test  = test[feature_cols],  test["label"]

print(f"Train: {len(train)} games\n seasons: {sorted(train['season'].unique())}")
print(f"Test:  {len(test)} games\n seasons: {sorted(test['season'].unique())}")
print(f"Train label mean: {y_train.mean():.3f}")
print(f"Test  label mean: {y_test.mean():.3f}")