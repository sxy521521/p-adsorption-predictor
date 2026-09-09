"""Bayesian optimization of XGBoost using only the 80% development set.

The held-out 20% test set is deliberately not used in this script.  Each Optuna
trial is evaluated by five-fold CV within the development set and is ranked by
mean RMSE.  The selected parameters are saved for all downstream scripts.
"""
from pathlib import Path
import json

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "adsorption_data_processed.csv"
RESULT_PATH = ROOT / "results" / "xgboost_bayesian_optimization_trials.csv"
PARAMETER_PATH = ROOT / "models" / "xgboost_best_params.json"
TARGET = "P adsorption capacity (mg/g)"
RANDOM_STATE = 42
N_TRIALS = 60


def main():
    data = pd.read_csv(DATA_PATH)
    x, y = data.drop(columns=[TARGET]), data[TARGET]
    x_dev, _, y_dev, _ = train_test_split(x, y, test_size=0.20, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.70, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        rmses, r2s = [], []
        for train_idx, valid_idx in cv.split(x_dev):
            model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, **params)
            model.fit(x_dev.iloc[train_idx], y_dev.iloc[train_idx])
            prediction = model.predict(x_dev.iloc[valid_idx])
            observed = y_dev.iloc[valid_idx]
            rmses.append(np.sqrt(mean_squared_error(observed, prediction)))
            r2s.append(r2_score(observed, prediction))
        trial.set_user_attr("cv_r2_mean", float(np.mean(r2s)))
        trial.set_user_attr("cv_r2_sd", float(np.std(r2s, ddof=1)))
        trial.set_user_attr("cv_rmse_sd", float(np.std(rmses, ddof=1)))
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="xgboost_phosphate_adsorption")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.rename(columns={"value": "cv_rmse_mean", "user_attrs_cv_r2_mean": "cv_r2_mean", "user_attrs_cv_r2_sd": "cv_r2_sd", "user_attrs_cv_rmse_sd": "cv_rmse_sd"}, inplace=True)
    RESULT_PATH.parent.mkdir(exist_ok=True)
    PARAMETER_PATH.parent.mkdir(exist_ok=True)
    trials.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    best = {key: int(value) if key in {"n_estimators", "max_depth", "min_child_weight"} else float(value) for key, value in study.best_params.items()}
    with PARAMETER_PATH.open("w", encoding="utf-8") as file:
        json.dump({"selection_method": "Optuna TPE Bayesian optimization; 5-fold CV on the 80% development set; objective = mean CV RMSE", "n_trials": N_TRIALS, "random_state": RANDOM_STATE, "best_cv_rmse": study.best_value, "best_cv_r2": study.best_trial.user_attrs["cv_r2_mean"], "parameters": best}, file, ensure_ascii=False, indent=2)

    print(json.dumps({"best_parameters": best, "best_cv_rmse": study.best_value, "best_cv_r2": study.best_trial.user_attrs["cv_r2_mean"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
