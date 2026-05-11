Environment
Python 3.9 or later is required. Install all dependencies with pip install pandas numpy scikit-learn xgboost statsmodels matplotlib.

Hardcoded paths
Every script in DataPrep/ has a project_base_path variable hardcoded to the original developer's machine. Update that variable at the top of each file to match your local path before running anything. Scripts in Modeling/ resolve their own directory automatically but also have a hardcoded data_path pointing to final_dataset.csv — update that too.

Data
Two data sources are already included in the repo. The Data/Kaggle Data/ folder contains tournament results, seeds, regular season results, and team name mappings from the March Machine Learning Mania Kaggle competition. The Data/20XX-20XX Team Results/ folders contain BartTorvik per-season adjusted efficiency stats (AdjOE, AdjDE, SOS) covering seasons 2008–2025, excluding 2020 when the tournament was cancelled.

Running the pipeline
Scripts must be run in order. Start with python DataPrep/build_features.py, which merges all data sources and writes DataPrep/final_dataset.csv. Then train each model: python Modeling/train_logistic.py, python Modeling/train_rf.py, and python Modeling/train_xgb.py, each of which writes a predictions CSV to Modeling/. Next run python Modeling/ensemble.py to average the three models' probabilities and write outputs to Results/. Finally, run mcnemar_test.py, get_feature_importance.py, calibration_plot.py, and report_outputs.py in any order to generate the remaining analysis files in Results/. The train/test split is seasons ≤ 2022 for training and seasons ≥ 2023 for testing.

