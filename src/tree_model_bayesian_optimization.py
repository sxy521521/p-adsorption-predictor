"""Tune CatBoost and LightGBM fairly with the same development-set protocol.

XGBoost is tuned in ``xgboost_bayesian_optimization.py``.  This script uses the
same 80% development set, five-fold CV, 60 TPE trials and mean CV RMSE objective
for CatBoost and LightGBM.  The independent 20% test set is never accessed.
"""
from pathlib import Path
import json

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import catboost as cb
import lightgbm as lgb

from model_preprocessing import TARGET, load_source_data, prepare_train_and_others


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
N_TRIALS = 60


def catboost_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 300, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
    }


def lightgbm_space(trial):
    max_depth = trial.suggest_int("max_depth", 3, 8)
    max_leaves = min(63, 2 ** max_depth)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 7, max_leaves),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def tune(model_name, space):
    data = load_source_data()
    development, _ = train_test_split(data, test_size=0.20, random_state=RANDOM_STATE)
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    prepared_folds = []
    for fold_index, (train_idx, valid_idx) in enumerate(folds.split(development)):
        prepared_train, [prepared_valid] = prepare_train_and_others(
            development.iloc[train_idx],
            [development.iloc[valid_idx]],
            random_state=RANDOM_STATE + fold_index,
        )
        prepared_folds.append((prepared_train, prepared_valid))

    def objective(trial):
        params = space(trial)
        rmses, r2s = [], []
        for prepared_train, prepared_valid in prepared_folds:
            x_train = prepared_train.drop(columns=[TARGET])
            y_train = prepared_train[TARGET]
            x_valid = prepared_valid.drop(columns=[TARGET])
            y_valid = prepared_valid[TARGET]
            if model_name == "CatBoost":
                model = cb.CatBoostRegressor(
                    loss_function="RMSE", verbose=False, random_seed=RANDOM_STATE,
                    thread_count=2, allow_writing_files=False, **params,
                )
            else:
                model = lgb.LGBMRegressor(
                    objective="regression", random_state=RANDOM_STATE, n_jobs=2,
                    verbosity=-1, **params,
                )
            model.fit(x_train, y_train)
            prediction = model.predict(x_valid)
            observed = y_valid
            rmses.append(np.sqrt(mean_squared_error(observed, prediction)))
            r2s.append(r2_score(observed, prediction))
        trial.set_user_attr("cv_r2_mean", float(np.mean(r2s)))
        trial.set_user_attr("cv_r2_sd", float(np.std(r2s, ddof=1)))
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"{model_name}_phosphate_adsorption")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    result = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    result.rename(columns={"value": "cv_rmse_mean", "user_attrs_cv_r2_mean": "cv_r2_mean", "user_attrs_cv_r2_sd": "cv_r2_sd"}, inplace=True)
    RESULT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    result.to_csv(RESULT_DIR / f"{model_name.lower()}_bayesian_optimization_trials.csv", index=False, encoding="utf-8-sig")
    best = {key: int(value) if key in {"iterations", "depth", "n_estimators", "max_depth", "num_leaves", "min_child_samples"} else float(value) for key, value in study.best_params.items()}
    # max_depth is sampled first in the LightGBM space and is therefore present;
    # num_leaves remains bounded by that sampled depth.
    with (MODEL_DIR / f"{model_name.lower()}_best_params.json").open("w", encoding="utf-8") as file:
        json.dump({"selection_method": "Optuna TPE Bayesian optimization; fold-wise DTR imputation and one-hot encoding; 5-fold CV on the 80% development set; objective = mean CV RMSE", "n_trials": N_TRIALS, "random_state": RANDOM_STATE, "best_cv_rmse": study.best_value, "best_cv_r2": study.best_trial.user_attrs["cv_r2_mean"], "parameters": best}, file, ensure_ascii=False, indent=2)
    print(model_name, json.dumps({"best_parameters": best, "best_cv_rmse": study.best_value, "best_cv_r2": study.best_trial.user_attrs["cv_r2_mean"]}, ensure_ascii=False))


def main():
    tune("CatBoost", catboost_space)
    tune("LightGBM", lightgbm_space)


if __name__ == "__main__":
    main()
