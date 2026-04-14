import pandas as pd

project_base_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project"
kaggle_data_path = project_base_path + "/Data/Kaggle Data"

# Load Kaggle Files
tourney_results = pd.read_csv(kaggle_data_path + "/MNCAATourneyCompactResults.csv")
tourney_seeds = pd.read_csv(kaggle_data_path + "/MNCAATourneySeeds.csv")
teams_df = pd.read_csv(kaggle_data_path + "/MTeams.csv")

# Filter to our seasons: 2008–2025, excluding 2020
our_seasons = list(range(2008, 2026))
our_seasons = [s for s in our_seasons if s != 2020]

tourney_results = tourney_results[tourney_results["Season"].isin(our_seasons)].copy()
tourney_seeds = tourney_seeds[tourney_seeds["Season"].isin(our_seasons)].copy()

# Extract numeric seed from seed string (e.g. "W01" → 1, "X16" → 16)
tourney_seeds["seed_num"] = tourney_seeds["Seed"].str.extract(r"(\d+)").astype(int)


# Verify shapes and seasons
print("tourney_results shape:", tourney_results.shape)
print("tourney_seeds shape:  ", tourney_seeds.shape)
print("\nSeasons in tourney_results:", sorted(tourney_results["Season"].unique()))
print("\nseed_num range:", tourney_seeds["seed_num"].min(), "to", tourney_seeds["seed_num"].max())

# Sample 5 rows with winner and loser team names joined in
sample = tourney_results.head(5).merge(
    teams_df[["TeamID", "TeamName"]].rename(columns={"TeamID": "WTeamID", "TeamName": "WTeamName"}),
    on="WTeamID"
).merge(
    teams_df[["TeamID", "TeamName"]].rename(columns={"TeamID": "LTeamID", "TeamName": "LTeamName"}),
    on="LTeamID"
)
print("\nSample games (winner vs loser):")
print(sample[["Season", "WTeamName", "WScore", "LTeamName", "LScore"]])