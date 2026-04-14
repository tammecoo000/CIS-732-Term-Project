import pandas as pd 
import glob
import os 
import warnings
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

project_base_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project"

# ── Load BartTorvik data (same as Step 1) 
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

# Load MTeamSpellings → build spelling lookup dict 
kaggle_data_path = project_base_path + "/Data/Kaggle Data"
team_spellings_df = pd.read_csv(kaggle_data_path + "/MTeamSpellings.csv")
spelling_to_id = dict(zip(team_spellings_df["TeamNameSpelling"], team_spellings_df["TeamID"]))

# Normalize BartTorvik names and attempt match 
barttorvik_all_seasons["team_lower"] = barttorvik_all_seasons["team"].str.lower().str.strip()
barttorvik_all_seasons["team_id"] = barttorvik_all_seasons["team_lower"].map(spelling_to_id)

# Report match results 
total_rows = len(barttorvik_all_seasons)
matched_rows = barttorvik_all_seasons["team_id"].notna().sum()
unmatched_rows = total_rows - matched_rows

print(f"Total rows: {total_rows}")
print(f"Matched: {matched_rows} ({matched_rows / total_rows * 100:.1f}%)")
print(f"Unmatched: {unmatched_rows}")

unmatched_teams = sorted(
    barttorvik_all_seasons.loc[barttorvik_all_seasons["team_id"].isna(), "team"].unique()
)
if unmatched_teams:
    print(f"\nUnmatched BartTorvik team names ({len(unmatched_teams)}):")
    for team_name in unmatched_teams:
        print(f"  '{team_name}'")
else:
    print("\nAll teams matched!")

# Manual corrections for the 15 unmatched names 
manual_corrections = {
    'arkansas pine bluff':    1115,
    'bethune cookman':        1126,
    'cal st. bakersfield':    1167,
    'illinois chicago':       1227,
    'louisiana monroe':       1419,
    'mississippi valley st.': 1290,
    'queens':                 1474,
    'saint francis':          1384,
    'southeast missouri st.': 1369,
    'st. francis ny':         1383,
    'tarleton st.':           1470,
    'tennessee martin':       1404,
    'texas a&m corpus chris': 1394,
    'ut rio grande valley':   1410,
    'winston salem st.':      1445,
}

still_unmatched_mask = barttorvik_all_seasons["team_id"].isna()
barttorvik_all_seasons.loc[still_unmatched_mask, "team_id"] = (
    barttorvik_all_seasons.loc[still_unmatched_mask, "team_lower"].map(manual_corrections)
)

final_unmatched = barttorvik_all_seasons["team_id"].isna().sum()
print(f"\nAfter manual corrections — still unmatched: {final_unmatched}")


# ── Keep only needed columns 
barttorvik_stats = barttorvik_all_seasons[["season", "team", "team_id", "adjoe", "adjde", "sos"]].copy()
print("\nFinal barttorvik_stats shape:", barttorvik_stats.shape)
print("\nSample:")
print(barttorvik_stats.head())