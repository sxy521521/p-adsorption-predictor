import csv
import json
import random
import math
from pathlib import Path

Path("results").mkdir(exist_ok=True)

def load_csv(filepath):
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def to_float(val):
    try:
        return float(val)
    except:
        return 0.0

def calc_mean(vals):
    return sum(vals) / len(vals) if vals else 0

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = [self.bias + sum(self.weights[i] * x[i] for i in range(n_features)) for x in X]

            dw = [0.0] * n_features
            db = 0
            for j in range(n_samples):
                error = y_pred[j] - y[j]
                db += error
                for i in range(n_features):
                    dw[i] += error * X[j][i]

            self.bias -= self.lr * db / n_samples
            for i in range(n_features):
                self.weights[i] -= self.lr * dw[i] / n_samples

    def predict(self, X):
        n_features = len(self.weights)
        return [self.bias + sum(self.weights[i] * x[i] for i in range(n_features)) for x in X]

def normalize_features(X_train):
    n_features = len(X_train[0])
    means = [calc_mean([X_train[i][j] for i in range(len(X_train))]) for j in range(n_features)]
    stds = [calc_mean([(X_train[i][j] - means[j])**2 for i in range(len(X_train))]) for j in range(n_features)]
    stds = [math.sqrt(s) for s in stds]

    for j in range(n_features):
        if stds[j] == 0:
            stds[j] = 1

    return means, stds

class GeneticAlgorithm:
    def __init__(self, n_variables, bounds, population_size=100, n_generations=30,
                 crossover_rate=0.7, mutation_rate=0.3):
        self.n_variables = n_variables
        self.bounds = bounds
        self.pop_size = population_size
        self.n_generations = n_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.population = []
        self.hall_of_fame = []
        self.pareto_front = []

    def create_individual(self):
        return [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.n_variables)]

    def create_population(self):
        self.population = [self.create_individual() for _ in range(self.pop_size)]

    def evaluate(self, individual):
        return [0, 0]

    def crossover(self, parent1, parent2):
        if random.random() < self.cx_rate:
            alpha = random.random()
            child1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(self.n_variables)]
            child2 = [(1 - alpha) * parent1[i] + alpha * parent2[i] for i in range(self.n_variables)]
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        for i in range(self.n_variables):
            if random.random() < self.mut_rate:
                delta = random.gauss(0, (self.bounds[i][1] - self.bounds[i][0]) * 0.1)
                individual[i] += delta
                individual[i] = max(self.bounds[i][0], min(self.bounds[i][1], individual[i]))
        return individual

    def select(self, population, fitnesses, k=2):
        selected = random.sample(list(zip(population, fitnesses)), k)
        return max(selected, key=lambda x: x[1])[0]

    def dominates(self, ind1, ind2):
        not_worse = all(f1 <= f2 for f1, f2 in zip(ind1['fitness'], ind2['fitness']))
        strictly_better = any(f1 < f2 for f1, f2 in zip(ind1['fitness'], ind2['fitness']))
        return not_worse and strictly_better

    def update_pareto_front(self, population_with_fitness):
        self.pareto_front = []
        for ind in population_with_fitness:
            is_dominated = False
            for other in population_with_fitness:
                if self.dominates(other, ind) and other != ind:
                    is_dominated = True
                    break
            if not is_dominated:
                self.pareto_front.append(ind)

    def evolve(self, model, means, stds, feature_cols):
        self.create_population()

        for gen in range(self.n_generations):
            pop_with_fitness = []
            for ind in self.population:
                ind_normalized = [(ind[i] - means[i]) / stds[i] for i in range(self.n_variables)]
                adsorption = model.predict([ind_normalized])[0]

                temp_idx = feature_cols.index('temperature') if 'temperature' in feature_cols else 0
                dosage_idx = feature_cols.index('adsorbent_dosage') if 'adsorbent_dosage' in feature_cols else 1
                energy = ind[temp_idx] * 0.5 + ind[dosage_idx] * 10

                ind_dict = {'genes': ind, 'fitness': [adsorption, -energy]}
                pop_with_fitness.append(ind_dict)

            self.update_pareto_front(pop_with_fitness)

            new_population = []
            while len(new_population) < self.pop_size:
                ind1 = random.choice(pop_with_fitness)['genes']
                ind2 = random.choice(pop_with_fitness)['genes']

                child1, child2 = self.crossover(ind1, ind2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            self.population = new_population[:self.pop_size]

            if gen % 5 == 0 or gen == self.n_generations - 1:
                best = max(pop_with_fitness, key=lambda x: x['fitness'][0])
                print(f"  Generation {gen+1:2d}: Best adsorption = {best['fitness'][0]:.2f}, Pareto size = {len(self.pareto_front)}")

        return self.pareto_front

def main():
    print("=== 遗传算法多目标优化 ===\n")

    rows = load_csv('data/processed/adsorption_data_processed.csv')
    fieldnames = list(rows[0].keys())
    feature_cols = [col for col in fieldnames if col != 'adsorption_capacity']

    X = [[to_float(row[col]) for col in feature_cols] for row in rows]
    y = [to_float(row['adsorption_capacity']) for row in rows]

    means, stds = normalize_features(X)
    X_norm = [[(X[i][j] - means[j]) / stds[j] for j in range(len(X[0]))] for i in range(len(X))]

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_norm, y)

    print("模型训练完成，开始GA优化...\n")

    bounds = []
    for i in range(len(feature_cols)):
        bounds.append((min([X[j][i] for j in range(len(X))]),
                      max([X[j][i] for j in range(len(X))])))

    ga = GeneticAlgorithm(
        n_variables=len(feature_cols),
        bounds=bounds,
        population_size=100,
        n_generations=30,
        crossover_rate=0.7,
        mutation_rate=0.3
    )

    print("优化进度:")
    pareto_solutions = ga.evolve(model, means, stds, feature_cols)

    print(f"\n找到 {len(pareto_solutions)} 个帕累托最优解")

    print("\n前10个最优解:")
    print(f"{'序号':<6} {'预测吸附容量':<18} {'能耗成本(相对)':<18} {'主要条件'}")
    print("-" * 80)

    sorted_pareto = sorted(pareto_solutions, key=lambda x: x['fitness'][0], reverse=True)[:10]
    for i, sol in enumerate(sorted_pareto, 1):
        genes = sol['genes']
        surface_idx = feature_cols.index('surface_area')
        ph_idx = feature_cols.index('pH')
        mod_idx = feature_cols.index('is_modified')

        condition = f"比表面积={genes[surface_idx]:.1f}, pH={genes[ph_idx]:.1f}, 改性={int(genes[mod_idx])}"

        print(f"{i:<6} {sol['fitness'][0]:<18.2f} {-sol['fitness'][1]:<18.2f} {condition}")

    pareto_data = []
    for sol in pareto_solutions:
        sol_dict = dict(zip(feature_cols, sol['genes']))
        sol_dict['predicted_adsorption'] = sol['fitness'][0]
        sol_dict['energy_cost'] = -sol['fitness'][1]
        pareto_data.append(sol_dict)

    with open('results/pareto_optimal_solutions.csv', 'w', newline='', encoding='utf-8-sig') as f:
        if pareto_data:
            writer = csv.DictWriter(f, fieldnames=list(pareto_data[0].keys()))
            writer.writeheader()
            writer.writerows(pareto_data)

    print("\n帕累托最优解已保存到 results/pareto_optimal_solutions.csv")

    print("\n" + "=" * 60)
    print("✓ 遗传算法优化完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
