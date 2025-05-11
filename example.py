import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# 1. Загрузка данных
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# 2. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Базовая точность
y_pred = model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# 5. Permutation Importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# 6. Сортировка и вывод
sorted_idx = result.importances_mean.argsort()[::-1]

print("\nPermutation Feature Importance:")
for i in sorted_idx[:10]:  # топ-10 признаков
    print(f"{feature_names[i]:<30} {result.importances_mean[i]:.4f}")

# 7. Визуализация
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx][:10][::-1], result.importances_mean[sorted_idx][:10][::-1])
plt.xlabel("Decrease in Accuracy")
plt.title("Top 10 Important Features (Permutation)")
plt.tight_layout()
plt.show()