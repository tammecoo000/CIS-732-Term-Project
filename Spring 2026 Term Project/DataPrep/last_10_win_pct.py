import pandas as pd

project_base_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project"
kaggle_data_path = project_base_path + "/Data/Kaggle Data"

# Load regular season results 
regular_season_df = pd.read_csv(kaggle_data_path + "/MRegularSeasonCompactResults.csv")

# Filter to our seasons: 2008–2025, excluding 2020
our_seasons = [s for s in range(2008, 2026) if s != 2020]
regular_season_df = regular_season_df[regular_season_df["Season"].isin(our_seasons)].copy()

# Reshape to long format: one row per team per game 
winner_rows = regular_season_df[["Season", "DayNum", "WTeamID"]].copy()
winner_rows.columns = ["season", "day_num", "team_id"]
winner_rows["won"] = 1

loser_rows = regular_season_df[["Season", "DayNum", "LTeamID"]].copy()
loser_rows.columns = ["season", "day_num", "team_id"]
loser_rows["won"] = 0

games_long = pd.concat([winner_rows, loser_rows], ignore_index=True)

# Compute last 10 games win percentage per (season, team_id) 
win_pct_last10 = (
    games_long.sort_values(["season", "team_id", "day_num"])
    .groupby(["season", "team_id"])["won"]
    .apply(lambda wins: wins.tail(10).mean())
    .reset_index(name="win_pct_last10")
)

win_pct_last10["win_pct_last10"] = win_pct_last10["win_pct_last10"].round(2)

#  Verify 
print("win_pct_last10 shape:", win_pct_last10.shape)
print("\nwin_pct_last10 range:", win_pct_last10["win_pct_last10"].min(), "to", win_pct_last10["win_pct_last10"].max())
print("\nNull values:", win_pct_last10["win_pct_last10"].isna().sum())
print("\nSample:")
print(win_pct_last10.head(10))
