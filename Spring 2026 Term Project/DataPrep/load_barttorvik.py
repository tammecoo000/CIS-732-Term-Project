import pandas as pd
import glob
import os 


barttorvik_folders = ["Data/2008-2014 Team Results", "Data/2015-2019 Team Results", "Data/2021-2026 Team Results"]

project_base_path = "/Users/coopertammen/Desktop/CIS732/Spring 2026 Term Project"

yearly_dataframes = []

for folder_name in barttorvik_folders:
    csv_search_pattern = os.path.join(project_base_path, folder_name, "*_team_results.csv")
    for csv_filepath in sorted(glob.glob(csv_search_pattern)):
        csv_filename = os.path.basename(csv_filepath)
        season_year = int(csv_filename.split("_")[0])
        if season_year == 2026:
            continue # not including 2026 results
        season_df = pd.read_csv(csv_filepath, index_col=False)
        season_df["season"] = season_year
        yearly_dataframes.append(season_df)


barttorvik_all_seasons = pd.concat(yearly_dataframes, ignore_index=True)


