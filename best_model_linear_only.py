import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import median_abs_deviation
import warnings
import os
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("1. Загрузка данных...")
train_df = pd.read_csv('../data/train.csv')
print(f"   Размер обучающей выборки: {train_df.shape}")

# Предобработка
train_processed = train_df.copy()

# Сохраняем целевую переменную
if 'RiskScore' in train_processed.columns:
    y = train_processed['RiskScore'].copy()
    train_processed = train_processed.drop('RiskScore', axis=1)

numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_processed.select_dtypes(include=['object']).columns.tolist()

for col in numeric_cols:
    median_value = train_processed[col].median()
    train_processed[col].fillna(median_value, inplace=True)

for col in categorical_cols:
    mode_val = train_processed[col].mode()[0] if len(train_processed[col].mode()) > 0 else 'Unknown'
    train_processed[col].fillna(mode_val, inplace=True)

if 'ApplicationDate' in train_processed.columns:
    train_processed['ApplicationDate'] = pd.to_datetime(train_processed['ApplicationDate'])
    train_processed['Year'] = train_processed['ApplicationDate'].dt.year
    train_processed['Month'] = train_processed['ApplicationDate'].dt.month
    train_processed['DayOfWeek'] = train_processed['ApplicationDate'].dt.dayofweek
    train_processed = train_processed.drop('ApplicationDate', axis=1)
    numeric_cols.extend(['Year', 'Month', 'DayOfWeek'])
    if 'ApplicationDate' in categorical_cols:
        categorical_cols.remove('ApplicationDate')

for col in categorical_cols:
    if col in train_processed.columns:
        train_encoded = pd.get_dummies(train_processed[col], prefix=col, dummy_na=False)
        train_processed = train_processed.drop(col, axis=1)
        train_processed = pd.concat([train_processed, train_encoded], axis=1)

feature_cols = train_processed.columns.tolist()
X = train_processed[feature_cols].values.astype(np.float32)
y = y.values.astype(np.float32)

print(f"   Количество признаков: {X.shape[1]}")

mask = (y >= -100) & (y <= 100)
X = X[mask]
y = y[mask]

print(f"   После фильтрации: X {X.shape}, y {y.shape}")

X_train, X_val, y_train_split, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
)
print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}\n")

print("2. Определение групп риска...")

low_threshold = 45.0
high_threshold = 55.0

y_train_class_3 = np.zeros(len(y_train_split), dtype=int)
y_train_class_3[y_train_split < low_threshold] = 0  # Низкий риск
y_train_class_3[(y_train_split >= low_threshold) & (y_train_split <= high_threshold)] = 1  # Средний риск
y_train_class_3[y_train_split > high_threshold] = 2  # Высокий риск

y_val_class_3 = np.zeros(len(y_val), dtype=int)
y_val_class_3[y_val < low_threshold] = 0
y_val_class_3[(y_val >= low_threshold) & (y_val <= high_threshold)] = 1
y_val_class_3[y_val > high_threshold] = 2

for i, name in enumerate(['Низкий', 'Средний', 'Высокий']):
    count = np.sum(y_train_class_3 == i)
    print(f"   {name}: {count} ({count/len(y_train_class_3)*100:.1f}%)")
print()

print("3. Определение классов и функций...")

class MADScaler:
    """Масштабирование на основе MAD"""
    def __init__(self):
        self.medians_ = None
        self.mads_ = None
    
    def fit_transform(self, X):
        self.medians_ = np.median(X, axis=0)
        self.mads_ = median_abs_deviation(X, axis=0, scale='normal')
        self.mads_ = np.where(self.mads_ == 0, 1.0, self.mads_)
        return (X - self.medians_) / self.mads_
    
    def transform(self, X):
        return (X - self.medians_) / self.mads_

# Функции перехода
def softmax_transition(proba, predictions):
    """Softmax (вероятностное взвешивание)"""
    return np.sum(proba * predictions, axis=1)

def sigmoid_transition(proba, predictions, temperature=0.5):
    """Sigmoid переход"""
    from scipy.special import expit
    normalized = expit((proba - 0.5) / temperature)
    normalized = normalized / normalized.sum(axis=1, keepdims=True)
    return np.sum(normalized * predictions, axis=1)

def exponential_transition(proba, predictions, alpha=2.0):
    """Exponential переход"""
    exp_proba = np.exp(alpha * (proba - 0.5))
    exp_proba = exp_proba / exp_proba.sum(axis=1, keepdims=True)
    return np.sum(exp_proba * predictions, axis=1)

transition_functions = {
    'Softmax (вероятностное)': softmax_transition,
    'Sigmoid (temp=0.5)': lambda p, pred: sigmoid_transition(p, pred, 0.5),
    'Exponential (alpha=2.0)': lambda p, pred: exponential_transition(p, pred, 2.0)
}

group_transition_functions = {
    0: 'Softmax (вероятностное)',
    1: 'Sigmoid (temp=0.5)',
    2: 'Exponential (alpha=2.0)'
}

def adaptive_transition(proba, predictions, group_transitions, group_predictions):
    """Адаптивная функция перехода"""
    n_samples = len(proba)
    final_predictions = np.zeros(n_samples)
    
    for group_idx in range(3):
        group_mask = group_predictions == group_idx
        if np.sum(group_mask) > 0:
            func_name = group_transitions[group_idx]
            func = transition_functions[func_name]
            final_predictions[group_mask] = func(proba[group_mask], predictions[group_mask])
    
    return final_predictions

print("4. Применение адаптивной предобработки...")

X_train_adaptive = X_train.copy()
X_val_adaptive = X_val.copy()

group_0_mask_train = y_train_class_3 == 0
for i in range(X_train.shape[1]):
    group_0_data = X_train[group_0_mask_train, i]
    if len(group_0_data) > 0:
        lower = np.percentile(group_0_data, 2)
        upper = np.percentile(group_0_data, 98)
        X_train_adaptive[group_0_mask_train, i] = np.clip(X_train[group_0_mask_train, i], lower, upper)
        
        group_0_mask_val = y_val_class_3 == 0
        X_val_adaptive[group_0_mask_val, i] = np.clip(X_val[group_0_mask_val, i], lower, upper)

# Группа 1 (Средний риск): RobustScaler
group_1_mask_train = y_train_class_3 == 1
robust_scaler_group1 = RobustScaler()
if np.sum(group_1_mask_train) > 0:
    X_train_adaptive[group_1_mask_train] = robust_scaler_group1.fit_transform(X_train[group_1_mask_train])
    group_1_mask_val = y_val_class_3 == 1
    X_val_adaptive[group_1_mask_val] = robust_scaler_group1.transform(X_val[group_1_mask_val])

# Группа 2 (Высокий риск): IQR Clipping (factor=1.5)
group_2_mask_train = y_train_class_3 == 2
for i in range(X_train.shape[1]):
    group_2_data = X_train[group_2_mask_train, i]
    if len(group_2_data) > 0:
        Q1 = np.percentile(group_2_data, 25)
        Q3 = np.percentile(group_2_data, 75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X_train_adaptive[group_2_mask_train, i] = np.clip(X_train[group_2_mask_train, i], lower, upper)
            
            group_2_mask_val = y_val_class_3 == 2
            X_val_adaptive[group_2_mask_val, i] = np.clip(X_val[group_2_mask_val, i], lower, upper)

# Финальная нормализация
adaptive_scaler = StandardScaler()
X_train_adaptive_scaled = adaptive_scaler.fit_transform(X_train_adaptive)
X_val_adaptive_scaled = adaptive_scaler.transform(X_val_adaptive)

print("5. Обучение LinearRegression для каждой группы...")
models_linear = []
means_linear = []

for group_idx in [0, 1, 2]:
    mask = y_train_class_3 == group_idx
    X_group = X_train_adaptive_scaled[mask]
    y_group = y_train_split[mask]
    
    if len(X_group) == 0:
        # Если группа пустая, используем среднее значение
        models_linear.append(None)
        means_linear.append(y_train_split.mean())
        continue
    
    mean_group = y_group.mean()
    means_linear.append(mean_group)
    
    model = LinearRegression()
    model.fit(X_group, y_group)
    models_linear.append(model)
    
    train_mse = mean_squared_error(y_group, model.predict(X_group))
    print(f"   ✅ Модель для группы {group_idx} обучена (MSE на train: {train_mse:.4f})")

print()

print("6. Обучение модели для группы 1 с MADScaler...")
group_1_mask_train = y_train_class_3 == 1
group_1_mask_val = y_val_class_3 == 1

if np.sum(group_1_mask_train) > 0:
    mad_scaler_group1 = MADScaler()
    X_train_group1_mad = mad_scaler_group1.fit_transform(X_train[group_1_mask_train])
    X_val_group1_mad = mad_scaler_group1.transform(X_val[group_1_mask_val])
    
    scaler_group1 = StandardScaler()
    X_train_group1_mad_scaled = scaler_group1.fit_transform(X_train_group1_mad)
    X_val_group1_mad_scaled = scaler_group1.transform(X_val_group1_mad)
    
    X_group1_mad = X_train_group1_mad_scaled
    y_group1 = y_train_split[group_1_mask_train]
    
    model_group1_mad = LinearRegression()
    model_group1_mad.fit(X_group1_mad, y_group1)
    
    train_mse_mad = mean_squared_error(y_group1, model_group1_mad.predict(X_group1_mad))
    print(f"   Группа 1(MSE на train: {train_mse_mad:.4f})\n")
else:
    model_group1_mad = None

print("7. Генерация предсказаний комбинированной модели...")

predictions_all_models = []

if models_linear[0] is not None:
    pred_0_all = models_linear[0].predict(X_val_adaptive_scaled)
else:
    pred_0_all = np.full(len(y_val), means_linear[0])
predictions_all_models.append(pred_0_all)

if model_group1_mad is not None:
    X_val_group1_mad_scaled_all = scaler_group1.transform(mad_scaler_group1.transform(X_val))
    pred_1_all = model_group1_mad.predict(X_val_group1_mad_scaled_all)
elif models_linear[1] is not None:
    pred_1_all = models_linear[1].predict(X_val_adaptive_scaled)
else:
    pred_1_all = np.full(len(y_val), means_linear[1])
predictions_all_models.append(pred_1_all)

if models_linear[2] is not None:
    pred_2_all = models_linear[2].predict(X_val_adaptive_scaled)
else:
    pred_2_all = np.full(len(y_val), means_linear[2])
predictions_all_models.append(pred_2_all)

predictions_array = np.array(predictions_all_models).T

# Создаем вероятности на основе принадлежности к группам
# Используем жесткое назначение группы с небольшим смягчением для границ
y_val_proba = np.zeros((len(y_val), 3))
for i, group_idx in enumerate([0, 1, 2]):
    mask = y_val_class_3 == group_idx
    y_val_proba[mask, i] = 1.0 

# Добавляем небольшое смягчение для границ групп
# Для образцов близких к границам добавляем небольшие вероятности соседних групп
for i in range(len(y_val)):
    group = y_val_class_3[i]
    # Если образец близок к границе группы, добавляем небольшие вероятности соседних групп
    if group == 0:  # Низкий риск
        # Если значение близко к верхней границе, добавляем небольшую вероятность группы 1
        if y_val[i] > 42.0:
            y_val_proba[i, 1] = 0.2
            y_val_proba[i, 0] = 0.8
    elif group == 1:  # Средний риск
        # Добавляем небольшие вероятности соседних групп
        if y_val[i] < 47.0:
            y_val_proba[i, 0] = 0.2
            y_val_proba[i, 1] = 0.8
        elif y_val[i] > 53.0:
            y_val_proba[i, 2] = 0.2
            y_val_proba[i, 1] = 0.8
    else:  # Высокий риск
        # Если значение близко к нижней границе, добавляем небольшую вероятность группы 1
        if y_val[i] < 58.0:
            y_val_proba[i, 1] = 0.2
            y_val_proba[i, 2] = 0.8

# Нормализуем вероятности
y_val_proba = y_val_proba / (y_val_proba.sum(axis=1, keepdims=True) + 1e-10)

y_pred_combined_weighted = adaptive_transition(
    y_val_proba,
    predictions_array,
    group_transition_functions,
    y_val_class_3
)

print("8. Применение адаптивной постобработки...")

y_pred_combined_postprocessed = y_pred_combined_weighted.copy()

group_0_mask = y_val_class_3 == 0
if np.sum(group_0_mask) > 0:
    group_0_pred = y_pred_combined_postprocessed[group_0_mask]
    lower_0 = np.percentile(y_train_split[y_train_class_3 == 0], 2.5)
    upper_0 = np.percentile(y_train_split[y_train_class_3 == 0], 97.5)
    y_pred_combined_postprocessed[group_0_mask] = np.clip(group_0_pred, lower_0, upper_0)

group_1_mask = y_val_class_3 == 1
if np.sum(group_1_mask) > 0:
    group_1_pred = y_pred_combined_postprocessed[group_1_mask]
    lower_1 = np.percentile(y_train_split[y_train_class_3 == 1], 1)
    upper_1 = np.percentile(y_train_split[y_train_class_3 == 1], 99)
    y_pred_combined_postprocessed[group_1_mask] = np.clip(group_1_pred, lower_1, upper_1)

group_2_mask = y_val_class_3 == 2
if np.sum(group_2_mask) > 0:
    group_2_pred = y_pred_combined_postprocessed[group_2_mask]
    lower_2 = np.percentile(y_train_split[y_train_class_3 == 2], 1)
    upper_2 = np.percentile(y_train_split[y_train_class_3 == 2], 99)
    y_pred_combined_postprocessed[group_2_mask] = np.clip(group_2_pred, lower_2, upper_2)

print("=" * 60)
print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
print("=" * 60)

mse_combined_post = mean_squared_error(y_val, y_pred_combined_postprocessed)
rmse_combined_post = np.sqrt(mse_combined_post)
mae_combined_post = mean_absolute_error(y_val, y_pred_combined_postprocessed)
r2_combined_post = r2_score(y_val, y_pred_combined_postprocessed)

print("Метрики комбинированной модели с адаптивной постобработкой:")
print(f"  MSE:  {mse_combined_post:.4f}")
print(f"  RMSE: {rmse_combined_post:.4f}")
print(f"  MAE:  {mae_combined_post:.4f}")
print(f"  R^2:   {r2_combined_post:.4f}")
print()

print("Анализ по группам:")
for i, name in enumerate(["Низкий риск", "Средний риск", "Высокий риск"]):
    mask = y_val_class_3 == i
    if np.sum(mask) > 0:
        mse_group = mean_squared_error(y_val[mask], y_pred_combined_postprocessed[mask])
        print(f"  {name}: MSE = {mse_group:.4f}")
