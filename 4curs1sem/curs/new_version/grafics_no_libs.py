import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Установка стиля графиков (Seaborn)
sns.set(style="whitegrid")

# =======================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =======================================

data_path = '../../data/csvdata.csv'
data = pd.read_csv(data_path)

# Определение признаков и целевой переменной
feature_columns = ['Area', 'No. of Bedrooms', 'City', 'Location']
target_column = 'Price'

# Сохраняем «до перемешивания» (для сравнения на графике)
data_before_shuffle = data.copy()

# Разделение данных
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

# Перемешивание данных
X, y = shuffle_dataset(X, y, random_state=42)

# Разделение на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split_manual(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split_manual(X_temp, y_temp, test_size=0.3333, random_state=42)

# Нормализация числовых данных
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

# Преобразование данных
def preprocess_data(X, scaler_info=None, encoder_info=None):
    X_transformed = pd.DataFrame()
    scaler_info = scaler_info or {}
    encoder_info = encoder_info or {}

    # Нормализация числовых данных
    for feature in numerical_features:
        scaled, min_val, max_val = min_max_scaler(
            X[feature],
            scaler_info.get(feature, {}).get('min'),
            scaler_info.get(feature, {}).get('max')
        )
        X_transformed[feature] = scaled
        scaler_info[feature] = {'min': min_val, 'max': max_val}

    # Кодирование категорий
    for feature in categorical_features:
        encoded, categories = encode_one_hot(
            X[feature],
            encoder_info.get(feature)
        )
        X_transformed = pd.concat([X_transformed, encoded], axis=1)
        encoder_info[feature] = categories

    return X_transformed, scaler_info, encoder_info

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    return obj

# =======================================
# 2. ВИЗУАЛИЗАЦИЯ «ПЕРЕМЕШИВАНИЯ»
# =======================================

# Для наглядности построим график «до» и «после» шифрования по индексу.
# Поскольку главная метрика - Price, покажем, как изменился порядок цен.

def plot_shuffle_comparison(original_df, shuffled_X, shuffled_y, save_path='shuffle_comparison.png'):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # График до перемешивания
    axes[0].plot(original_df.index, original_df[target_column], 'bo-', markersize=3)
    axes[0].set_title("До перемешивания (индекс vs Price)")
    axes[0].set_xlabel("Индекс (до shuffle)")
    axes[0].set_ylabel("Price")

    # График после перемешивания
    axes[1].plot(shuffled_X.index, shuffled_y, 'ro-', markersize=3)
    axes[1].set_title("После перемешивания (индекс vs Price)")
    axes[1].set_xlabel("Индекс (после shuffle)")
    axes[1].set_ylabel("Price")

    plt.tight_layout()
    plt.savefig(save_path)  # Сохраняем в PNG
    plt.close(fig)

# Вызываем визуализацию «перемешивания»
plot_shuffle_comparison(data_before_shuffle, X, y, save_path='shuffle_comparison.png')

# =======================================
# 3. ПРЕОБРАЗОВАНИЕ ДАННЫХ (НОРМАЛИЗАЦИЯ)
# =======================================

X_train_processed, scaler_info, encoder_info = preprocess_data(X_train)
X_val_processed, _, _ = preprocess_data(X_val, scaler_info, encoder_info)
X_test_processed, _, _ = preprocess_data(X_test, scaler_info, encoder_info)

# Визуализируем нормализацию на примере обучающей выборки:
# Сравним распределения числовых признаков «до» и «после» Min-Max.

def plot_normalization_comparison(X_before, X_after, numeric_feats, save_path='normalization_comparison.png'):
    # Построим гистограммы (или KDE) для каждого признака до/после
    fig, axes = plt.subplots(nrows=len(numeric_feats), ncols=2, figsize=(12, 5 * len(numeric_feats)))

    for i, feature in enumerate(numeric_feats):
        # До нормализации
        axes[i, 0].hist(X_before[feature], bins=30, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f'{feature} - ДО нормализации')
        axes[i, 0].set_xlabel(feature)
        axes[i, 0].set_ylabel("Количество")

        # После нормализации
        axes[i, 1].hist(X_after[feature], bins=30, color='salmon', edgecolor='black')
        axes[i, 1].set_title(f'{feature} - ПОСЛЕ нормализации')
        axes[i, 1].set_xlabel(feature)
        axes[i, 1].set_ylabel("Количество")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# Вызываем визуализацию «до» и «после» нормализации
plot_normalization_comparison(X_train, X_train_processed, numerical_features, save_path='normalization_comparison.png')

# =======================================
# 4. РЕАЛИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ
# =======================================

class SimpleRandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=5, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        for _ in range(self.n_trees):
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            tree = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) <= 1:
            return y.mean()
        feature = np.random.choice(X.columns)
        threshold = X[feature].median()
        left_indices = X[feature] <= threshold
        right_indices = X[feature] > threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return {"feature": feature, "threshold": threshold, "left": left_tree, "right": right_tree}

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += self._predict_tree(tree, X)
        return predictions / len(self.trees)

    def _predict_tree(self, tree, X):
        if isinstance(tree, float):
            return np.full(len(X), tree)
        feature, threshold = tree["feature"], tree["threshold"]
        left_indices = X[feature] <= threshold
        right_indices = X[feature] > threshold
        predictions = np.zeros(len(X))
        predictions[left_indices] = self._predict_tree(tree["left"], X[left_indices])
        predictions[right_indices] = self._predict_tree(tree["right"], X[right_indices])
        return predictions

# Обучение модели
model = SimpleRandomForestRegressor(n_trees=10, max_depth=5, random_state=42)
model.fit(X_train_processed, y_train)

# Оценка модели
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

val_predictions = model.predict(X_val_processed)
val_rmse = rmse(y_val, val_predictions)

test_predictions = model.predict(X_test_processed)
test_rmse = rmse(y_test, test_predictions)

# Сохранение модели
with open('simple_random_forest.json', 'w') as f:
    json.dump(make_serializable({
        "model": model.trees,
        "scaler_info": scaler_info,
        "encoder_info": encoder_info
    }), f)

# Преобразование результатов для отображения
results = {
    "Validation RMSE": float(val_rmse),
    "Test RMSE": float(test_rmse)
}

# Итоговый вывод результатов
print("Model Performance:", results)
print("Графики сохранены в файлы 'shuffle_comparison.png' и 'normalization_comparison.png'.")
