import pandas as pd
import numpy as np
import glob 
import os 
import warnings
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

project_base_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project"
kaggle_data_path = project_base_path + "/Data/Kaggle Data"
our_seasons = [s for s in range(2008, 2026) if s != 2020]

# Load Tournament Results
tourney_results= pd.read_csv(kaggle_data_path + "/MNCAATourneyCompactResults.csv")
tourney_results = tourney_results[tourney_results["Season"].isin(our_seasons)].copy()
tourney_results = tourney_results.rename(columns={"Season": "season", "DayNum": "day_num", "WTeamID": "w_team_id", "LTeamID": "l_team_id"})

# Load tournament seeds
tourney_seeds = pd.read_csv(kaggle_data_path + "/MNCAATourneySeeds.csv")
tourney_seeds = tourney_seeds[tourney_seeds["Season"].isin(our_seasons)].copy()
tourney_seeds = tourney_seeds.rename(columns={"Season": "season", "TeamID": "team_id"})
tourney_seeds["seed_num"] = tourney_seeds["Seed"].str.extract(r"(\d+)").astype(int)

# Load BartTorvik stats with team_id (Steps 1 + 3) 
barttorvik_folders = [
    "Data/2008-2014 Team Results",
    "Data/2015-2019 Team Results",
    "Data/2021-2026 Team Results",
]
yearly_dataframes = []
for folder_name in barttorvik_folders:
    csv_search_pattern = os.path.join(project_base_path, folder_name, "*_team_results.csv")
    for csv_filepath in sorted(glob.glob(csv_search_pattern)):
        csv_filename = os.path.basename(csv_filepath)
        season_year = int(csv_filename.split("_")[0])
        if season_year == 2026:
            continue
        season_df = pd.read_csv(csv_filepath, index_col=False)
        season_df["season"] = season_year
        yearly_dataframes.append(season_df)
barttorvik_all_seasons = pd.concat(yearly_dataframes, ignore_index=True)

team_spellings_df = pd.read_csv(kaggle_data_path + "/MTeamSpellings.csv")
spelling_to_id = dict(zip(team_spellings_df["TeamNameSpelling"], team_spellings_df["TeamID"]))
barttorvik_all_seasons["team_lower"] = barttorvik_all_seasons["team"].str.lower().str.strip()
barttorvik_all_seasons["team_id"] = barttorvik_all_seasons["team_lower"].map(spelling_to_id)

manual_corrections = {
    'arkansas pine bluff': 1115, 'bethune cookman': 1126,
    'cal st. bakersfield': 1167, 'illinois chicago': 1227,
    'louisiana monroe': 1419, 'mississippi valley st.': 1290,
    'queens': 1474, 'saint francis': 1384,
    'southeast missouri st.': 1369, 'st. francis ny': 1383,
    'tarleton st.': 1470, 'tennessee martin': 1404,
    'texas a&m corpus chris': 1394, 'ut rio grande valley': 1410,
    'winston salem st.': 1445,
}
unmatched_mask = barttorvik_all_seasons["team_id"].isna()
barttorvik_all_seasons.loc[unmatched_mask, "team_id"] = (
    barttorvik_all_seasons.loc[unmatched_mask, "team_lower"].map(manual_corrections)
)
barttorvik_stats = barttorvik_all_seasons[["season", "team_id", "adjoe", "adjde", "sos"]].copy()
barttorvik_stats["team_id"] = barttorvik_stats["team_id"].astype(int)

# Load last 10 win percentage (Step 4) 
regular_season_df = pd.read_csv(kaggle_data_path + "/MRegularSeasonCompactResults.csv")
regular_season_df = regular_season_df[regular_season_df["Season"].isin(our_seasons)].copy()

winner_rows = regular_season_df[["Season", "DayNum", "WTeamID"]].rename(
    columns={"Season": "season", "DayNum": "day_num", "WTeamID": "team_id"})
winner_rows["won"] = 1
loser_rows = regular_season_df[["Season", "DayNum", "LTeamID"]].rename(
    columns={"Season": "season", "DayNum": "day_num", "LTeamID": "team_id"})
loser_rows["won"] = 0
games_long = pd.concat([winner_rows, loser_rows], ignore_index=True)
win_pct_last10 = (
    games_long.sort_values(["season", "team_id", "day_num"])
    .groupby(["season", "team_id"])["won"]
    .apply(lambda wins: wins.tail(10).mean())
    .reset_index(name="win_pct_last10")
)
win_pct_last10["win_pct_last10"] = win_pct_last10["win_pct_last10"].round(2)

# Prepare lookup tables 
seed_lookup = tourney_seeds[["season", "team_id", "seed_num"]]
bt_lookup = barttorvik_stats[["season", "team_id", "adjoe", "adjde", "sos"]]
l10_lookup = win_pct_last10[["season", "team_id", "win_pct_last10"]]

# Join all features onto tournament games 
games_df = tourney_results[["season", "day_num", "w_team_id", "l_team_id"]].copy()

games_df = games_df.merge(
    seed_lookup.rename(columns={"team_id": "w_team_id", "seed_num": "w_seed"}),
    on=["season", "w_team_id"], how="left")
games_df = games_df.merge(
    seed_lookup.rename(columns={"team_id": "l_team_id", "seed_num": "l_seed"}),
    on=["season", "l_team_id"], how="left")
games_df = games_df.merge(
    bt_lookup.rename(columns={"team_id": "w_team_id", "adjoe": "w_adjoe", "adjde": "w_adjde", "sos": "w_sos"}),
    on=["season", "w_team_id"], how="left")
games_df = games_df.merge(
    bt_lookup.rename(columns={"team_id": "l_team_id", "adjoe": "l_adjoe", "adjde": "l_adjde", "sos": "l_sos"}),
    on=["season", "l_team_id"], how="left")
games_df = games_df.merge(
    l10_lookup.rename(columns={"team_id": "w_team_id", "win_pct_last10": "w_win_pct_last10"}),
    on=["season", "w_team_id"], how="left")
games_df = games_df.merge(
    l10_lookup.rename(columns={"team_id": "l_team_id", "win_pct_last10": "l_win_pct_last10"}),
    on=["season", "l_team_id"], how="left")

# Map DayNum to tournament round 
day_to_round = {
    134: 0, 135: 0,
    136: 1, 137: 1,
    138: 2, 139: 2, 140: 2,   
    143: 3, 144: 3,
    145: 4, 146: 4,
    147: 4, 148: 4,           
    152: 5, 153: 5,
    154: 6, 155: 6,
}

games_df["round"] = games_df["day_num"].map(day_to_round)

# Randomly assign Team A / Team B
np.random.seed(42)
winner_is_team_a = np.random.randint(0, 2, size=len(games_df)).astype(bool)

w_cols = ["w_team_id", "w_seed", "w_adjoe", "w_adjde", "w_sos", "w_win_pct_last10"]
l_cols = ["l_team_id", "l_seed", "l_adjoe", "l_adjde", "l_sos", "l_win_pct_last10"]

for w_col, l_col in zip(w_cols, l_cols):
    base = w_col[2:]
    games_df[f"a_{base}"] = np.where(winner_is_team_a, games_df[w_col], games_df[l_col])
    games_df[f"b_{base}"] = np.where(winner_is_team_a, games_df[l_col], games_df[w_col])

games_df["label"] = winner_is_team_a.astype(int)

# Compute difference features
games_df["delta_adjoe"] = games_df["a_adjoe"] - games_df["b_adjoe"]
games_df["delta_adjde"] = games_df["b_adjde"] - games_df["a_adjde"]
games_df["delta_sos"] = games_df["a_sos"] - games_df["b_sos"]
games_df["seed_gap"] = games_df["b_seed"] - games_df["a_seed"]
games_df["delta_win_pct_last10"] = games_df["a_win_pct_last10"] - games_df["b_win_pct_last10"]

# Final dataset 
feature_columns = [
    "season", "round",
    "delta_adjoe", "delta_adjde", "delta_sos",
    "seed_gap", "delta_win_pct_last10", "label"
]

pre_drop = games_df[feature_columns]
print(f"\nRows before dropna: {len(pre_drop)}")
print("Null counts per column before dropna:")
print(pre_drop.isnull().sum())
final_dataset = pre_drop.dropna().reset_index(drop=True)
print(f"Rows after dropna:  {len(final_dataset)}  (dropped {len(pre_drop) - len(final_dataset)})")


print("Final dataset shape:", final_dataset.shape)
final_dataset.to_csv("final_dataset.csv", index=False)
print("Saved to final_dataset.csv")
print("\nLabel distribution:")
print(final_dataset["label"].value_counts())
print("\nNull values per column:")
print(final_dataset.isnull().sum())
print("\nFeature ranges:")
print(final_dataset.describe().round(3))



# Step 6: Spot-check with real team names 
teams_df = pd.read_csv(kaggle_data_path + "/MTeams.csv")
team_name_lookup = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

games_df["w_team_name"] = games_df["w_team_id"].map(team_name_lookup)
games_df["l_team_name"] = games_df["l_team_id"].map(team_name_lookup)

spot_check = games_df[["season", "round", "w_team_name", "w_seed", "l_team_name", "l_seed"]].copy()
spot_check.columns = ["season", "round", "winner", "winner_seed", "loser", "loser_seed"]
spot_check["upset"] = spot_check["winner_seed"] > spot_check["loser_seed"]

# Print 10 sample games
print("\nSample games (winner vs loser):")
print(spot_check.head(10).to_string(index=False))

# Spot-check: 2018 UMBC over Virginia (only 16-over-1 upset in history)
umbc_upset = spot_check[
    (spot_check["season"] == 2018) &
    (spot_check["winner_seed"] == 16) &
    (spot_check["loser_seed"] == 1)
]
print("\n2018 16-over-1 upsets (expect UMBC over Virginia):")
print(umbc_upset.to_string(index=False))

# Spot-check: 2023 Fairleigh Dickinson over Purdue (16 over 1)
fdu_upset = spot_check[
    (spot_check["season"] == 2023) &
    (spot_check["winner_seed"] == 16) &
    (spot_check["loser_seed"] == 1)
]
print("\n2023 16-over-1 upsets (expect FDU over Purdue):")
print(fdu_upset.to_string(index=False))

# How many upsets total (winner was the higher seed number)?
total_upsets = spot_check["upset"].sum()
print(f"\nTotal upsets in dataset (higher seed beat lower seed): {total_upsets} / {len(spot_check)}")
