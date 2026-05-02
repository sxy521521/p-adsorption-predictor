import csv
import json
import random
import math
from pathlib import Path
from collections import defaultdict

Path("models").mkdir(exist_ok=True)
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

def train_test_split_data(rows, test_size=0.2, random_seed=42):
    random.seed(random_seed)
    shuffled = rows.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_idx], shuffled[split_idx:]

def calc_mean(vals):
    return sum(vals) / len(vals) if vals else 0

def calc_std(vals, mean):
    variance = sum((v - mean)**2 for v in vals) / len(vals)
    return math.sqrt(variance)

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

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _best_split(self, X, y):
        best_gain = -float('inf')
        best_feature = 0
        best_threshold = 0

        n_samples, n_features = len(X), len(X[0])
        for feature in range(n_features):
            thresholds = set(x[feature] for x in X)
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = [], [], [], []
                for i in range(n_samples):
                    if X[i][feature] <= threshold:
                        left_X.append(X[i])
                        left_y.append(y[i])
                    else:
                        right_X.append(X[i])
                        right_y.append(y[i])

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = self._variance_reduction(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _variance_reduction(self, y, left_y, right_y):
        var_parent = calc_std(y, calc_mean(y)) ** 2
        n = len(y)
        var_left = calc_std(left_y, calc_mean(left_y)) ** 2 * len(left_y) / n
        var_right = calc_std(right_y, calc_mean(right_y)) ** 2 * len(right_y) / n
        return var_parent - var_left - var_right

    def _build_tree(self, X, y, depth):
        n_samples = len(X)
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'leaf': True, 'value': calc_mean(y)}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'value': calc_mean(y)}

        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(n_samples):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    def _predict_sample(self, x, node):
        if node.get('leaf', False):
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        return self._predict_sample(x, node['right'])

    def predict(self, X):
        return [self._predict_sample(x, self.tree) for x in X]

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y, n_features_select=None):
        n_samples = len(X)
        n_features = len(X[0]) if n_features_select is None else min(n_features_select, len(X[0]))

        for _ in range(self.n_trees):
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_bootstrap = [X[i] for i in indices]
            y_bootstrap = [y[i] for i in indices]

            feature_indices = random.sample(range(len(X[0])), n_features)
            X_subset = [[x[i] for i in feature_indices] for x in X_bootstrap]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y_bootstrap)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        predictions = []
        for tree, feature_indices in self.trees:
            X_subset = [[x[i] for i in feature_indices] for x in X]
            predictions.append(tree.predict(X_subset))

        n_samples = len(X)
        return [calc_mean([predictions[t][i] for t in range(self.n_trees)]) for i in range(n_samples)]

def r2_score(y_true, y_pred):
    ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    ss_tot = sum((y_true[i] - calc_mean(y_true))**2 for i in range(len(y_true)))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def rmse(y_true, y_pred):
    n = len(y_true)
    return math.sqrt(sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n)

def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

def normalize_features(X_train, X_test):
    n_features = len(X_train[0])
    means = [calc_mean([X_train[i][j] for i in range(len(X_train))]) for j in range(n_features)]
    stds = [calc_std([X_train[i][j] for i in range(len(X_train))], means[j]) for j in range(n_features)]

    for j in range(n_features):
        if stds[j] == 0:
            stds[j] = 1

    X_train_norm = [[(X_train[i][j] - means[j]) / stds[j] for j in range(n_features)] for i in range(len(X_train))]
    X_test_norm = [[(X_test[i][j] - means[j]) / stds[j] for j in range(n_features)] for i in range(len(X_test))]

    return X_train_norm, X_test_norm, means, stds

def main():
    print("=== 机器学习模型训练 ===\n")

    rows = load_csv('data/processed/adsorption_data_processed.csv')
    fieldnames = list(rows[0].keys())
    feature_cols = [col for col in fieldnames if col != 'adsorption_capacity']

    X = [[to_float(row[col]) for col in feature_cols] for row in rows]
    y = [to_float(row['adsorption_capacity']) for row in rows]

    X_train_raw, X_test_raw = train_test_split_data(list(zip(X, y)), test_size=0.2, random_seed=42)
    X_train = [x for x, _ in X_train_raw]
    y_train = [v for _, v in X_train_raw]
    X_test = [x for x, _ in X_test_raw]
    y_test = [v for _, v in X_test_raw]

    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"特征数: {len(feature_cols)}")

    X_train_norm, X_test_norm, means, stds = normalize_features(X_train, X_test)

    models = {}

    print("\n=== 训练线性回归 ===")
    lr_model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    lr_model.fit(X_train_norm, y_train)
    lr_pred = lr_model.predict(X_test_norm)
    lr_metrics = {
        'R²': r2_score(y_test, lr_pred),
        'RMSE': rmse(y_test, lr_pred),
        'MAE': mae(y_test, lr_pred)
    }
    models['LinearRegression'] = {'metrics': lr_metrics, 'model': lr_model}
    print(f"线性回归 R²: {lr_metrics['R²']:.4f}")

    print("\n=== 训练决策树 ===")
    dt_model = DecisionTree(max_depth=10, min_samples_split=5)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_metrics = {
        'R²': r2_score(y_test, dt_pred),
        'RMSE': rmse(y_test, dt_pred),
        'MAE': mae(y_test, dt_pred)
    }
    models['DecisionTree'] = {'metrics': dt_metrics, 'model': dt_model}
    print(f"决策树 R²: {dt_metrics['R²']:.4f}")

    print("\n=== 训练随机森林 ===")
    rf_model = RandomForest(n_trees=20, max_depth=10, min_samples_split=5)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        'R²': r2_score(y_test, rf_pred),
        'RMSE': rmse(y_test, rf_pred),
        'MAE': mae(y_test, rf_pred)
    }
    models['RandomForest'] = {'metrics': rf_metrics, 'model': rf_model}
    print(f"随机森林 R²: {rf_metrics['R²']:.4f}")

    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)
    print(f"\n{'模型':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 52)
    for name, data in models.items():
        m = data['metrics']
        print(f"{name:<20} {m['R²']:>10.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f}")

    best_model_name = max(models.keys(), key=lambda k: models[k]['metrics']['R²'])
    best_model = models[best_model_name]['model']
    best_metrics = models[best_model_name]['metrics']

    print(f"\n最佳模型: {best_model_name} (R² = {best_metrics['R²']:.4f})")

    model_info = {
        'best_model': best_model_name,
        'metrics': best_metrics,
        'feature_names': feature_cols,
        'normalization': {'means': means, 'stds': stds}
    }
    with open('models/model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"\n模型信息已保存到 models/model_info.json")

    feature_importance = {}
    if best_model_name == 'RandomForest':
        for tree, feat_indices in best_model.trees:
            for i, feat_idx in enumerate(feat_indices):
                feat_name = feature_cols[feat_idx]
                if feat_name not in feature_importance:
                    feature_importance[feat_name] = 0
                feature_importance[feat_name] += 1
    elif best_model_name == 'DecisionTree':
        def count_features(node, counts):
            if node.get('leaf'):
                return
            feat_name = feature_cols[node['feature']]
            counts[feat_name] = counts.get(feat_name, 0) + 1
            count_features(node['left'], counts)
            count_features(node['right'], counts)
        count_features(best_model.tree, feature_importance)

    if feature_importance:
        print("\n特征重要性:")
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feat, count in sorted_importance[:10]:
            print(f"  {feat}: {count}")

        feature_importance_data = [{'feature': f, 'importance': c} for f, c in sorted_importance]
        with open('results/feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump(feature_importance_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✓ 模型训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
