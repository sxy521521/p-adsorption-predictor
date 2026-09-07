"""用 NSGA-II 型多目标遗传算法优化 P 吸附容量与能耗代理指标，并绘制图 7。"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "adsorption_data_processed.csv"
MODEL_PATH = PROJECT_DIR / "models" / "best_XGBoost_model.pkl"
FIGURE_PATH = PROJECT_DIR / "results" / "figures" / "07_xgboost_multiobjective_ga.png"
SUMMARY_PATH = PROJECT_DIR / "results" / "ga_optimization_summary.csv"
CONVERGENCE_PATH = PROJECT_DIR / "results" / "ga_optimization_convergence.csv"
SOLUTIONS_PATH = PROJECT_DIR / "results" / "ga_optimization_solutions.csv"

TARGET = "P adsorption capacity (mg/g)"
STATUS_COL = "Modified or unmodified"
CROSSLINK_COL = "Cross-linked or uncross-linked"
INITIAL_P_COL = "Initial P concentration (mg/L)"
TEMPERATURE_COL = "Reactor temperature (℃)"
TIME_COL = "Reaction time (min)"
CONTINUOUS_COLS = [
    "Adsorbent dosage (g/L) ",
    TEMPERATURE_COL,
    INITIAL_P_COL,
    TIME_COL,
    "Solution pH",
    "Pore volume (cm³/g)",
    "BET surface area (m²/g)",
]
MATERIAL_PREFIX = "Modified material type_"
AGENT_PREFIX = "Cross-linking agent type_"

POPULATION_SIZE = 80
GENERATIONS = 500
RUNS_PER_SCENARIO = 10
MUTATION_RATE = 0.10
MUTATION_SCALE = 0.10
RANDOM_STATE = 42
# 1149.12 mg/L 在原始数据中仅出现一次；优化仅在有充分数据支持的实际范围内进行。
INITIAL_P_OPTIMISATION_MAX = 500.0


def non_dominated_fronts(objectives: np.ndarray) -> list[np.ndarray]:
    """返回最小化问题的非支配层，目标数为 2。"""
    remaining = np.arange(len(objectives))
    fronts = []
    while len(remaining):
        values = objectives[remaining]
        dominates = (
            (values[:, None, :] <= values[None, :, :]).all(axis=2)
            & (values[:, None, :] < values[None, :, :]).any(axis=2)
        )
        is_dominated = dominates.any(axis=0)
        front = remaining[~is_dominated]
        fronts.append(front)
        remaining = remaining[is_dominated]
    return fronts


def crowding_distance(objectives: np.ndarray, front: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(front))
    if len(front) <= 2:
        distance[:] = np.inf
        return distance
    values = objectives[front]
    for objective_index in range(values.shape[1]):
        order = np.argsort(values[:, objective_index])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = values[order[-1], objective_index] - values[order[0], objective_index]
        if span > 0:
            distance[order[1:-1]] += (
                values[order[2:], objective_index] - values[order[:-2], objective_index]
            ) / span
    return distance


def nsga2_metadata(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    ranks = np.empty(len(objectives), dtype=int)
    distances = np.zeros(len(objectives))
    fronts = non_dominated_fronts(objectives)
    for rank, front in enumerate(fronts):
        ranks[front] = rank
        distances[front] = crowding_distance(objectives, front)
    return ranks, distances, fronts


def nsga2_select(objectives: np.ndarray, population_size: int) -> np.ndarray:
    _, _, fronts = nsga2_metadata(objectives)
    selected = []
    for front in fronts:
        remaining_slots = population_size - len(selected)
        if len(front) <= remaining_slots:
            selected.extend(front.tolist())
        else:
            distances = crowding_distance(objectives, front)
            selected.extend(front[np.argsort(-distances)[:remaining_slots]].tolist())
            break
    return np.asarray(selected, dtype=int)


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.75)
    ax.grid(False)
    ax.tick_params(labelsize=8.2, width=0.7, length=3, color="#303030")


class XGBoostGAOptimizer:
    """在实测边界内，以 NSGA-II 同时优化预测 P 吸附量与温度-时间能耗。"""

    def __init__(self, x: pd.DataFrame, y: pd.Series, model):
        self.x = x
        self.model = model
        self.model_columns = list(model.feature_names_in_)
        self.continuous_bounds = np.array([
            [float(x[column].min()), float(x[column].max())] for column in CONTINUOUS_COLS
        ])
        self.target_min = float(y.min())
        self.target_span = float(y.max() - y.min())
        self.temperature_index = CONTINUOUS_COLS.index(TEMPERATURE_COL)
        self.initial_p_index = CONTINUOUS_COLS.index(INITIAL_P_COL)
        self.time_index = CONTINUOUS_COLS.index(TIME_COL)
        self.continuous_bounds[self.initial_p_index, 1] = min(
            self.continuous_bounds[self.initial_p_index, 1], INITIAL_P_OPTIMISATION_MAX
        )
        self.material_columns = [
            column for column in self.model_columns if column.startswith(MATERIAL_PREFIX)
        ]
        self.agent_columns = [
            column for column in self.model_columns if column.startswith(AGENT_PREFIX)
        ]
        if not self.material_columns:
            raise ValueError("Modified material type columns were not found.")
        if not self.agent_columns:
            raise ValueError("Cross-linking agent type columns were not found.")
        # 仅保留原始数据中真实出现过的“改性/交联/材料/交联剂”组合，
        # 避免优化过程虚构类别组合。
        self.category_columns = self.material_columns + self.agent_columns
        self.valid_combinations = x[
            [STATUS_COL, CROSSLINK_COL, *self.category_columns]
        ].drop_duplicates(ignore_index=True)
        self.column_index = {column: index for index, column in enumerate(self.model_columns)}

    def repair(self, population: np.ndarray, fixed_initial_p: float | None) -> np.ndarray:
        repaired = population.copy()
        repaired[:, 0] = np.clip(np.rint(repaired[:, 0]), 0, len(self.valid_combinations) - 1)
        repaired[:, 1:] = np.clip(
            repaired[:, 1:], self.continuous_bounds[:, 0], self.continuous_bounds[:, 1]
        )
        if fixed_initial_p is not None:
            repaired[:, 1 + self.initial_p_index] = fixed_initial_p
        return repaired

    def initialise(self, rng: np.random.Generator, fixed_initial_p: float | None) -> np.ndarray:
        population = np.empty((POPULATION_SIZE, 1 + len(CONTINUOUS_COLS)))
        population[:, 0] = rng.integers(0, len(self.valid_combinations), size=POPULATION_SIZE)
        lower, upper = self.continuous_bounds[:, 0], self.continuous_bounds[:, 1]
        population[:, 1:] = rng.uniform(lower, upper, size=(POPULATION_SIZE, len(CONTINUOUS_COLS)))
        return self.repair(population, fixed_initial_p)

    def model_input(self, population: np.ndarray) -> pd.DataFrame:
        matrix = np.zeros((len(population), len(self.model_columns)), dtype=float)
        combinations = self.valid_combinations.iloc[np.rint(population[:, 0]).astype(int)]
        matrix[:, self.column_index[STATUS_COL]] = combinations[STATUS_COL].to_numpy()
        matrix[:, self.column_index[CROSSLINK_COL]] = combinations[CROSSLINK_COL].to_numpy()
        for decision_index, column in enumerate(CONTINUOUS_COLS):
            matrix[:, self.column_index[column]] = population[:, 1 + decision_index]
        for column in self.category_columns:
            matrix[:, self.column_index[column]] = combinations[column].to_numpy(dtype=float)
        return pd.DataFrame(matrix, columns=self.model_columns)

    def evaluate(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = np.asarray(self.model.predict(self.model_input(population)))
        normalised_prediction = (prediction - self.target_min) / self.target_span
        temperature = population[:, 1 + self.temperature_index]
        reaction_time = population[:, 1 + self.time_index]
        temperature_norm = (
            (temperature - self.continuous_bounds[self.temperature_index, 0])
            / (self.continuous_bounds[self.temperature_index, 1] - self.continuous_bounds[self.temperature_index, 0])
        )
        time_norm = (
            (reaction_time - self.continuous_bounds[self.time_index, 0])
            / (self.continuous_bounds[self.time_index, 1] - self.continuous_bounds[self.time_index, 0])
        )
        energy_proxy = temperature_norm * time_norm
        # NSGA-II 的两个最小化目标：最大 P 吸附量（取负）与最小能耗代理值。
        objectives = np.column_stack([-normalised_prediction, energy_proxy])
        # 与原文 3.4 的组合目标一致，仅用于展示每代最优进展及选择折衷解。
        composite_objective = -normalised_prediction + energy_proxy
        return prediction, objectives, composite_objective

    def offspring(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        rng: np.random.Generator,
        fixed_initial_p: float | None,
    ) -> np.ndarray:
        ranks, distances, _ = nsga2_metadata(objectives)

        def tournament() -> int:
            a, b = rng.integers(0, len(population), size=2)
            if ranks[a] != ranks[b]:
                return a if ranks[a] < ranks[b] else b
            return a if distances[a] >= distances[b] else b

        child = np.empty_like(population)
        for row in range(len(population)):
            parent_a, parent_b = population[tournament()], population[tournament()]
            child[row, 0] = parent_a[0] if rng.random() < 0.5 else parent_b[0]
            blend = rng.random(len(CONTINUOUS_COLS))
            child[row, 1:] = blend * parent_a[1:] + (1 - blend) * parent_b[1:]

        continuous_mutation = rng.random(child[:, 1:].shape) < MUTATION_RATE
        mutation_noise = rng.normal(
            0,
            MUTATION_SCALE * (self.continuous_bounds[:, 1] - self.continuous_bounds[:, 0]),
            size=child[:, 1:].shape,
        )
        child[:, 1:] += continuous_mutation * mutation_noise
        category_mutation = rng.random(len(child)) < MUTATION_RATE
        child[category_mutation, 0] = rng.integers(0, len(self.valid_combinations), size=category_mutation.sum())
        return self.repair(child, fixed_initial_p)

    def run(self, fixed_initial_p: float | None, seed: int) -> tuple[dict, np.ndarray]:
        rng = np.random.default_rng(seed)
        population = self.initialise(rng, fixed_initial_p)
        prediction, objectives, composite = self.evaluate(population)
        history = np.empty(GENERATIONS)
        for generation in range(GENERATIONS):
            children = self.offspring(population, objectives, rng, fixed_initial_p)
            child_prediction, child_objectives, child_composite = self.evaluate(children)
            combined_population = np.vstack([population, children])
            combined_prediction = np.concatenate([prediction, child_prediction])
            combined_objectives = np.vstack([objectives, child_objectives])
            combined_composite = np.concatenate([composite, child_composite])
            selected = nsga2_select(combined_objectives, POPULATION_SIZE)
            population = combined_population[selected]
            prediction = combined_prediction[selected]
            objectives = combined_objectives[selected]
            composite = combined_composite[selected]
            history[generation] = composite.min()

        best_index = int(np.argmin(composite))
        solution = {"predicted_p_adsorption_capacity_mg_g": float(prediction[best_index]),
                    "energy_proxy": float(objectives[best_index, 1]),
                    "composite_objective": float(composite[best_index]),
                    "combination_code": int(round(population[best_index, 0]))}
        combination = self.valid_combinations.iloc[solution["combination_code"]]
        solution[STATUS_COL] = combination[STATUS_COL]
        solution[CROSSLINK_COL] = combination[CROSSLINK_COL]
        material_types = [
            column.replace(MATERIAL_PREFIX, "") for column in self.material_columns
            if combination[column] == 1
        ]
        agent_types = [
            column.replace(AGENT_PREFIX, "") for column in self.agent_columns
            if combination[column] == 1
        ]
        solution["modified_material_type"] = material_types[0] if material_types else "baseline"
        solution["cross_linking_agent_type"] = agent_types[0] if agent_types else "0"
        for decision_index, column in enumerate(CONTINUOUS_COLS):
            solution[column.strip()] = float(population[best_index, 1 + decision_index])
        return solution, history


def main():
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    model = joblib.load(MODEL_PATH)
    x = x.loc[:, model.feature_names_in_]
    optimizer = XGBoostGAOptimizer(x, y, model)

    initial_p_values = x.loc[
        x[INITIAL_P_COL] <= INITIAL_P_OPTIMISATION_MAX, INITIAL_P_COL
    ].to_numpy()
    scenarios = []
    for percentile in range(10, 101, 10):
        scenarios.append((f"{percentile}th percentile", float(np.percentile(initial_p_values, percentile))))
    scenarios.append(("Practical range (≤500 mg/L)", None))

    solution_rows = []
    convergence_rows = []
    summary_rows = []
    histories_by_scenario = {}
    for scenario_index, (scenario_label, fixed_initial_p) in enumerate(scenarios):
        scenario_histories = []
        scenario_solutions = []
        for run in range(RUNS_PER_SCENARIO):
            solution, history = optimizer.run(
                fixed_initial_p=fixed_initial_p,
                seed=RANDOM_STATE + scenario_index * 1000 + run,
            )
            solution.update({
                "scenario": scenario_label,
                "initial_p_constraint_mg_l": fixed_initial_p if fixed_initial_p is not None else "Practical range (≤500 mg/L)",
                "run": run + 1,
            })
            solution_rows.append(solution)
            scenario_solutions.append(solution)
            scenario_histories.append(history)

        histories = np.asarray(scenario_histories)
        histories_by_scenario[scenario_label] = histories
        for generation in range(GENERATIONS):
            values = histories[:, generation]
            convergence_rows.append({
                "scenario": scenario_label,
                "generation": generation + 1,
                "objective_mean": float(values.mean()),
                "objective_lower_95": float(np.percentile(values, 2.5)),
                "objective_upper_95": float(np.percentile(values, 97.5)),
            })
        predicted_values = np.array([item["predicted_p_adsorption_capacity_mg_g"] for item in scenario_solutions])
        energy_values = np.array([item["energy_proxy"] for item in scenario_solutions])
        summary_rows.append({
            "scenario": scenario_label,
            "initial_p_constraint_mg_l": fixed_initial_p if fixed_initial_p is not None else "Practical range (≤500 mg/L)",
            "predicted_p_mean_mg_g": float(predicted_values.mean()),
            "predicted_p_lower_95_mg_g": float(np.percentile(predicted_values, 2.5)),
            "predicted_p_upper_95_mg_g": float(np.percentile(predicted_values, 97.5)),
            "energy_proxy_mean": float(energy_values.mean()),
        })
        print(f"Completed: {scenario_label}")

    summary = pd.DataFrame(summary_rows)
    solutions = pd.DataFrame(solution_rows)
    convergence = pd.DataFrame(convergence_rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    solutions.to_csv(SOLUTIONS_PATH, index=False, encoding="utf-8-sig")
    convergence.to_csv(CONVERGENCE_PATH, index=False, encoding="utf-8-sig")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, (ax_response, ax_convergence) = plt.subplots(
        1, 2, figsize=(11.4, 4.65), gridspec_kw={"width_ratios": [0.90, 1.55]}
    )
    x_position = np.arange(len(summary))
    response_color = "#C73E1D"
    ax_response.fill_between(
        x_position,
        summary["predicted_p_lower_95_mg_g"],
        summary["predicted_p_upper_95_mg_g"],
        color="#E8B2A6", alpha=0.48, linewidth=0,
    )
    ax_response.plot(
        x_position, summary["predicted_p_mean_mg_g"], color=response_color,
        linewidth=1.35, marker="o", markersize=3.3,
    )
    ax_response.set_xticks(x_position)
    ax_response.set_xticklabels(
        [label.replace(" percentile", "th") if label != "Full range" else label for label in summary["scenario"]],
        rotation=65, ha="right", fontsize=7.2,
    )
    ax_response.set_xlabel("Initial P concentration constraint (≤500 mg/L)", fontsize=9.3)
    ax_response.set_ylabel("Predicted P adsorption capacity (mg/g)", fontsize=9.3)
    style_axis(ax_response)

    colors = [
        "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
        "#6A3D9A", "#A6761D", "#1B9E77", "#17BECF", "#222222",
    ]
    generation = np.arange(1, GENERATIONS + 1)
    for color, scenario_label in zip(colors, summary["scenario"]):
        values = histories_by_scenario[scenario_label]
        mean = values.mean(axis=0)
        lower = np.percentile(values, 2.5, axis=0)
        upper = np.percentile(values, 97.5, axis=0)
        ax_convergence.fill_between(generation, lower, upper, color=color, alpha=0.12, linewidth=0)
        ax_convergence.plot(generation, mean, color=color, linewidth=0.85, label=scenario_label)
    ax_convergence.set_xlabel("Iterations", fontsize=9.3)
    ax_convergence.set_ylabel("Composite objective value", fontsize=9.3)
    ax_convergence.legend(
        loc="upper right", ncol=2, fontsize=6.1, frameon=True, facecolor="white",
        edgecolor="#B7B7B7", framealpha=0.92, borderpad=0.45, handlelength=1.4,
    )
    style_axis(ax_convergence)
    fig.tight_layout(w_pad=2.0, pad=0.9)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved convergence: {CONVERGENCE_PATH}")
    print(f"Saved solutions: {SOLUTIONS_PATH}")


if __name__ == "__main__":
    main()
