"""Tune CatBoost strictly within the proper-training subset used by ICP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from model_preprocessing import TARGET, load_source_data, prepare_train_and_others


ROOT = Path(__file__).resolve().parent.parent
PARAMETER_PATH = ROOT / "models" / "catboost_icp_best_params.json"
RESULT_PATH = ROOT / "results" / "catboost_icp_bayesian_optimization_trials.csv"
RANDOM_STATE = 42
N_TRIALS = 60


def parameter_space(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "iterations": trial.suggest_int("iterations", 300, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
    }


def main() -> None:
    source = load_source_data()
    development, _ = train_test_split(source, test_size=0.20, random_state=RANDOM_STATE)
    proper_train, _ = train_test_split(development, test_size=0.20, random_state=RANDOM_STATE)
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    prepared_folds = []
    for fold_index, (train_idx, valid_idx) in enumerate(folds.split(proper_train)):
        train, [valid] = prepare_train_and_others(
            proper_train.iloc[train_idx],
            [proper_train.iloc[valid_idx]],
            random_state=RANDOM_STATE + fold_index,
        )
        prepared_folds.append((train, valid))

    def objective(trial: optuna.Trial) -> float:
        params = parameter_space(trial)
        rmses, r2s = [], []
        for train, valid in prepared_folds:
            model = CatBoostRegressor(
                loss_function="RMSE", verbose=False, random_seed=RANDOM_STATE,
                thread_count=2, allow_writing_files=False, **params,
            )
            x_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
            x_valid, y_valid = valid.drop(columns=[TARGET]), valid[TARGET]
            model.fit(x_train, y_train)
            prediction = model.predict(x_valid)
            rmses.append(np.sqrt(mean_squared_error(y_valid, prediction)))
            r2s.append(r2_score(y_valid, prediction))
        trial.set_user_attr("cv_r2_mean", float(np.mean(r2s)))
        return float(np.mean(rmses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True),
        study_name="catboost_icp_phosphate_adsorption",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    RESULT_PATH.parent.mkdir(exist_ok=True)
    PARAMETER_PATH.parent.mkdir(exist_ok=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.rename(columns={"value": "cv_rmse_mean", "user_attrs_cv_r2_mean": "cv_r2_mean"}, inplace=True)
    trials.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    best = {key: int(value) if key in {"iterations", "depth"} else float(value) for key, value in study.best_params.items()}
    with PARAMETER_PATH.open("w", encoding="utf-8") as file:
        json.dump({"selection_method": "Optuna TPE Bayesian optimization; fold-wise DTR imputation and one-hot encoding; 5-fold CV on the 64% proper-training set used by ICP", "n_trials": N_TRIALS, "random_state": RANDOM_STATE, "best_cv_rmse": study.best_value, "best_cv_r2": study.best_trial.user_attrs["cv_r2_mean"], "parameters": best}, file, ensure_ascii=False, indent=2)
    print(json.dumps({"best_parameters": best, "best_cv_rmse": study.best_value}, ensure_ascii=False))


if __name__ == "__main__":
    main()
