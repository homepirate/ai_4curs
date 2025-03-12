import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns  # <-- ДОБАВЛЕНО для стильного графика

# ---------- 1. ЗАГРУЗКА ДАННЫХ И ФУНКЦИИ ДЛЯ ОЦЕНКИ МОДЕЛИ ----------

data_path = '../../data/csvdata.csv'
data = pd.read_csv(data_path)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Определение признаков и целевой переменной
feature_columns = ['Area', 'No. of Bedrooms', 'City', 'Location']
target_column = 'Price'

# Разделение данных на X (признаки) и y (целевая переменная)
X = data[feature_columns]
y = data[target_column]

def shuffle_dataset(X, y, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)

def train_test_split_manual(X, y, test_size, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_index = int(len(X) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

# Перемешивание (shuffle)
X, y = shuffle_dataset(X, y, random_state=42)

# Разделение на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split_manual(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split_manual(X_temp, y_temp, test_size=0.3333, random_state=42)

# ---------- 2. НОРМАЛИЗАЦИЯ И КОДИРОВАНИЕ ----------

numerical_features = ['Area', 'No. of Bedrooms']
categorical_features = ['City', 'Location']

def min_max_scaler(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val

def encode_one_hot(data, categories=None):
    if categories is None:
        categories = sorted(data.unique())
    encoding = {cat: (data == cat).astype(int) for cat in categories}
    return pd.DataFrame(encoding), categories

def preprocess_data(X, scaler_info=None, encoder_info=None):
    """ Масштабирование числовых признаков и One-Hot кодирование категориальных. """
    X_transformed = pd.DataFrame()
    scaler_info = scaler_info or {}
    encoder_info = encoder_info or {}

    # Масштабирование числовых
    for feature in numerical_features:
        scaled, min_val, max_val = min_max_scaler(
            X[feature],
            scaler_info.get(feature, {}).get('min'),
            scaler_info.get(feature, {}).get('max')
        )
        X_transformed[feature] = scaled
        scaler_info[feature] = {'min': min_val, 'max': max_val}

    # Кодирование категориальных
    for feature in categorical_features:
        encoded, categories = encode_one_hot(
            X[feature],
            encoder_info.get(feature)
        )
        X_transformed = pd.concat([X_transformed, encoded], axis=1)
        encoder_info[feature] = categories

    return X_transformed, scaler_info, encoder_info

def make_serializable(obj):
    """ Преобразование в сериализуемый вид (float, int, list, dict). """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    return obj

# Преобразуем обучающую, валидационную, тестовую
X_train_processed, scaler_info, encoder_info = preprocess_data(X_train)
X_val_processed, _, _ = preprocess_data(X_val, scaler_info, encoder_info)
X_test_processed, _, _ = preprocess_data(X_test, scaler_info, encoder_info)

# ---------- 3. УПРОЩЁННЫЙ СЛУЧАЙНЫЙ ЛЕС, ВЫБОР ПРИЗНАКОВ И ВАЛИД. ОШИБКА ----------

class SimpleRandomForestRegressor:
    def __init__(self, n_trees=50, max_depth=5, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        # Будем хранить только Validation RMSE
        self.val_errors = []

    def fit(self, X_train, y_train, X_val=None, y_val=None, track_error=False):
        """
        После обучения каждого дерева вычисляем RMSE на валидации,
        чтобы построить одну кривую (Validation RMSE).
        """
        np.random.seed(self.random_state)
        self.trees = []
        n_features = X_train.shape[1]

        for i in range(self.n_trees):
            # 1. Бутстреп-выборка
            sample_indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]

            # 2. Строим дерево (добавим случайный выбор признаков для разнообразия)
            #    Например, возьмём половину признаков случайно
            #    Если хотите больше/меньше - меняйте пропорцию.
            n_sub = max(1, n_features // 2)  # половина признаков, минимум 1
            features_subset = np.random.choice(X_train.columns, size=n_sub, replace=False)

            tree = self._build_tree(X_sample[features_subset], y_sample, depth=0, used_features=features_subset)
            self.trees.append((tree, features_subset))

            # 3. Если track_error=True, вычисляем RMSE на валидации
            if track_error and X_val is not None and y_val is not None:
                val_pred = self._predict_with_n_trees(X_val, n_trees=len(self.trees))
                val_rmse_ = rmse(y_val, val_pred)
                self.val_errors.append(val_rmse_)

    def _predict_with_n_trees(self, X, n_trees):
        """ Предсказать, используя первые n_trees деревьев. """
        predictions = np.zeros(len(X))
        for i in range(n_trees):
            tree, f_sub = self.trees[i]
            predictions += self._predict_tree(tree, X[f_sub])
        return predictions / n_trees

    def _build_tree(self, X, y, depth, used_features):
        # Условия остановки
        if depth >= self.max_depth or len(y) <= 1:
            return y.mean()

        # Случайный выбор признака из уже отобранных used_features
        feature = np.random.choice(used_features)
        # Порог = медиана
        threshold = X[feature].median()

        # Разделяем
        left_indices = X[feature] <= threshold
        right_indices = X[feature] > threshold

        # Рекурсивно строим поддеревья
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1, used_features)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1, used_features)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def predict(self, X):
        """
        Предсказание на всех деревьях (усреднение).
        """
        predictions = np.zeros(len(X))
        for (tree, f_sub) in self.trees:
            predictions += self._predict_tree(tree, X[f_sub])
        return predictions / len(self.trees)

    def _predict_tree(self, tree, X):
        # Если это число (float), значит мы в листе
        if isinstance(tree, float):
            return np.full(len(X), tree)

        feature = tree["feature"]
        threshold = tree["threshold"]

        left_indices = X[feature] <= threshold
        right_indices = X[feature] > threshold

        predictions = np.zeros(len(X))
        predictions[left_indices] = self._predict_tree(tree["left"], X[left_indices])
        predictions[right_indices] = self._predict_tree(tree["right"], X[right_indices])
        return predictions

# ---------- 4. ПОСТРОЕНИЕ ОДНОЙ КРИВОЙ (ВАЛИДАЦИОННАЯ ОШИБКА) ----------

def plot_validation_curve(model, save_path='learning_curve_single.png'):
    """
    Рисуем ОДНУ кривую:
    - Ось X: количество деревьев (1..n_trees)
    - Ось Y: Validation RMSE
    """
    # Используем стили Seaborn для красоты
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    x_axis = range(1, len(model.val_errors) + 1)
    # Можно вместо raw-значений сделать сглаживание (rolling mean)
    # Но здесь сначала просто нарисуем «как есть».
    sns.lineplot(x=x_axis, y=model.val_errors, marker='o', color='red', lw=2, markers=True, markersize=8)

    plt.xlabel('Число деревьев')
    plt.ylabel('Validation RMSE')
    plt.title('Validation Error by Number of Trees')
    plt.grid(True)

    # Сохраняем
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

# ---------- 5. ОБУЧАЕМ МОДЕЛЬ, ПОЛУЧАЕМ ОДНУ КРИВУЮ ОШИБКИ ----------

model = SimpleRandomForestRegressor(n_trees=50, max_depth=5, random_state=42)

# Указываем track_error=True и передаём валидационные данные
model.fit(X_train_processed, y_train, X_val_processed, y_val, track_error=True)

# Строим и сохраняем одиночную кривую
plot_validation_curve(model, save_path='learning_curve_single.png')
print("График (одна кривая Validation RMSE) сохранён в 'learning_curve_single.png'.")

# Вычисляем финальные метрики
val_predictions = model.predict(X_val_processed)
val_rmse_value = rmse(y_val, val_predictions)

test_predictions = model.predict(X_test_processed)
test_rmse_value = rmse(y_test, test_predictions)

results = {
    "Validation RMSE": float(val_rmse_value),
    "Test RMSE": float(test_rmse_value)
}
print("Model Performance:", results)

# ---------- 6. СОХРАНЕНИЕ МОДЕЛИ И ИНФОРМАЦИИ О СКЕЙЛИНГЕ ----------

with open('simple_random_forest.json', 'w') as f:
    json.dump(
        make_serializable({
            "model": [ (str(t[0]), list(t[1])) for t in model.trees ],  # Упрощённый вид для JSON
            "scaler_info": scaler_info,
            "encoder_info": encoder_info
        }),
        f
    )

print("Модель сохранена в 'simple_random_forest.json'.")
