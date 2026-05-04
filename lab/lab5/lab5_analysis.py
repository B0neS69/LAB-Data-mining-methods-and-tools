import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

print("=== ЛАБОРАТОРНА РОБОТА 5: АНАЛІЗ ТА ШАБЛОНИ ===\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ЕТАП 1: Підготовка даних
print("Етап 1: Завантаження та очищення даних")
csv_path = os.path.join(BASE_DIR, 'credit_risk_dataset.csv')

try:
    data = pd.read_csv(csv_path).dropna()
    print(f"Дані успішно завантажено. Робоча кількість записів: {len(data)}")
except FileNotFoundError:
    print(f"ПОМИЛКА: Файл '{csv_path}' не знайдено!")
    exit()

# ЕТАП 2: Кластерний аналіз (K-Means)
print("\nЕтап 2: Кластерний аналіз")
# Обираємо ознаки: вік, річний дохід та сума кредиту
features = data[['person_age', 'person_income', 'loan_amnt']]

# Стандартизація даних (дуже важливо для K-Means, щоб дохід у тисячах не перекривав вік у десятках)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 1. Пошук оптимальної кількості кластерів (Метод Ліктя)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
plt.title('Метод Ліктя (Elbow Method) для визначення кількості кластерів')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Сума квадратів відстаней (WCSS)')
plt.grid(True)
elbow_path = os.path.join(BASE_DIR, 'elbow_plot.png')
plt.savefig(elbow_path)
print(f"Графік Методу Ліктя збережено як '{elbow_path}'")

# 2. Фінальна кластеризація (використаємо k=4 на основі візуального зламу 'ліктя')
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans_final.fit_predict(features_scaled)

print(f"\nСтворено {optimal_k} кластери. Характеристики (середні значення):")
cluster_summary = data.groupby('Cluster')[['person_age', 'person_income', 'loan_amnt']].mean().round(1)
print(cluster_summary)

# ЕТАП 3: Крос-табуляція (Cross-tabulation)
print("\nЕтап 3: Крос-табуляція")
# Аналізуємо зв'язок між кредитним рейтингом (loan_grade) та статусом дефолту (loan_status)
crosstab_abs = pd.crosstab(data['loan_grade'], data['loan_status'])
print("1. Абсолютні значення (Кількість кредитів за рейтингом):")
print(crosstab_abs)

crosstab_pct = pd.crosstab(data['loan_grade'], data['loan_status'], normalize='index') * 100
print("\n2. Відносні значення (% ризикових кредитів у кожному рейтингу):")
print(crosstab_pct.round(1))

# Візуалізація крос-табуляції
crosstab_pct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#2ca02c', '#d62728'])
plt.title('Крос-табуляція: Ризик дефолту залежно від кредитного рейтингу')
plt.xlabel('Кредитний рейтинг (Loan Grade)')
plt.ylabel('Відсоток кредитів (%)')
plt.legend(['0 - Повернуто', '1 - Дефолт'], loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
crosstab_path = os.path.join(BASE_DIR, 'crosstab_plot.png')
plt.savefig(crosstab_path)
print(f"Графік крос-табуляції збережено як '{crosstab_path}'")

#ЕТАП 4: Дистиляція шаблонів даних
print("\nЕтап 4: Дистиляція шаблонів даних")
# Підготовка: перетворюємо текстові змінні на числа
X = pd.get_dummies(data.drop('loan_status', axis=1), drop_first=True)
y = data['loan_status']

# Використовуємо Дерево рішень з обмеженою глибиною для вилучення зрозумілих "шаблонів" (правил)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

print("Витягнуті базові шаблони (Логічні правила системи):")
tree_rules = export_text(dt, feature_names=list(X.columns))
print(tree_rules)

# Визначення найважливіших ознак у знайдених шаблонах
importances = dt.feature_importances_
feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(5)

print("\nТоп-5 факторів, що формують ці шаблони (Важливість ознак):")
print(feature_imp.round(3))

# Візуалізація важливості ознак
plt.figure(figsize=(9, 5))
feature_imp.plot(kind='barh', color='indigo')
plt.title('Дистиляція шаблонів: Топ-5 найвпливовіших ознак')
plt.xlabel('Вага ознаки в шаблоні')
plt.gca().invert_yaxis() 
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
patterns_path = os.path.join(BASE_DIR, 'patterns_plot.png')
plt.savefig(patterns_path)
print(f"Графік шаблонів збережено як '{patterns_path}'")