import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import networkx as nx
from networkx.algorithms import community

print("=== ЗАПУСК АНАЛІЗУ ДАНИХ (DATA MINING) ===\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ЕТАП 1: Завантаження та підготовка даних
print("Етап 1: Підготовка та очищення РЕАЛЬНИХ даних")
csv_path = os.path.join(BASE_DIR, 'credit_risk_dataset.csv')

try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"ПОМИЛКА: Файл '{csv_path}' не знайдено!")
    print("Будь ласка, переконайтеся, що ви завантажили датасет і поклали його в одну папку зі скриптом.")
    exit()

# Очищення від порожніх рядків
data_cleaned = data.dropna()
print(f"Початкова кількість записів: {len(data)}. Після очищення: {len(data_cleaned)}")

# ЕТАП 2: Дерево рішень 
print("\nЕтап 2: Побудова Дерева рішень")
# Перетворення текстових категорій на числа (кодування)
data_encoded = pd.get_dummies(data_cleaned, drop_first=True)

# Розділяємо дані на вхідні ознаки (X) та цільову змінну (y)
X = data_encoded.drop('loan_status', axis=1)
y = data_encoded['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Оцінка ефективності моделі на тестовій вибірці:")
print(f"Точність (Accuracy): {accuracy_score(y_test, y_pred):.3f}")
print(f"Чутливість (Recall): {recall_score(y_test, y_pred, pos_label=1):.3f}")
print(f"Точність передбачення (Precision): {precision_score(y_test, y_pred, pos_label=1):.3f}")

# ЕТАП 3: K-Means Кластеризація
print("\nЕтап 3: Кластеризація K-середніх (K-Means)")
# Вибираємо колонки віку та доходу
X_cluster = data_cleaned[['person_age', 'person_income']].copy()
# Фільтруємо екстремальні викиди для адекватного графіка
X_cluster = X_cluster[(X_cluster['person_age'] < 80) & (X_cluster['person_income'] < 200000)]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_cluster)
centroids = kmeans.cluster_centers_

print("Аналіз отриманих кластерів (Центроїди):")
for i, centroid in enumerate(centroids):
    print(f"Кластер {i}: Середній вік ~ {int(centroid[0])} років, Середній дохід ~ {int(centroid[1])} $")

plt.figure(figsize=(10, 6))
plt.scatter(X_cluster['person_age'], X_cluster['person_income'], c=labels, cmap='viridis', alpha=0.5, label='Клієнти')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, linewidths=3, label='Центроїди')
plt.title('Кластеризація клієнтів (Вік vs Дохід)')
plt.xlabel('Вік (років)')
plt.ylabel('Річний дохід ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

kmeans_path = os.path.join(BASE_DIR, 'kmeans_plot.png')
plt.savefig(kmeans_path)
print(f"Графік збережено як '{kmeans_path}'")

# ЕТАП 4: Асоціативні правила
print("\nЕтап 4: Асоціативні правила")
# Беремо категорії: тип житла та мета кредиту
cat_data = data_cleaned[['person_home_ownership', 'loan_intent']]
basket = pd.get_dummies(cat_data).astype(bool)

freq_items = apriori(basket, min_support=0.1, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)

print("Знайдені асоціативні правила (Топ-3):")
top_rules = rules.sort_values('support', ascending=False).head(3)
for idx, row in top_rules.iterrows():
    antecedents = list(row['antecedents'])[0]
    consequents = list(row['consequents'])[0]
    print(f"Якщо {antecedents} => То {consequents} (Впевненість: {row['confidence']:.2f})")

# ЕТАП 5: Байєсівська класифікація
print("\nЕтап 5: Байєсівська класифікація текстів")
train_texts = [

    "Як правильно інвестувати гроші в акції та державні облігації",
    "Банк пропонує вигідні умови для відкриття нового депозиту",
    "Курс іноземних валют різко змінився після новин з біржі",
    "Оформлення кредиту на житло з низькою відсотковою ставкою",
    "Інфляція та її негативний вплив на світову економіку",
    "Як розрахувати податки для малого та середнього бізнесу",
    "Аналітика криптовалютного ринку та прогноз ціни біткоїна",

    "Розробка сучасних веб-додатків за допомогою React та Node.js",
    "Як виправити критичну помилку компіляції в коді мовою C++",
    "Налаштування та оптимізація сервера бази даних PostgreSQL",
    "Штучний інтелект генерує реалістичні зображення за текстом",
    "Оптимізація швидкості завантаження сайту для пошукових систем",
    "Використання Docker контейнерів для розгортання мікросервісів",
    "Основи об'єктно-орієнтованого програмування та патерни"
]
train_labels = ["Finance", "Finance", "Finance", "Finance", "Finance", "Finance", "Finance", 
                "IT", "IT", "IT", "IT", "IT", "IT", "IT"]

vectorizer = CountVectorizer()
X_train_text = vectorizer.fit_transform(train_texts)

nb_model = MultinomialNB()
nb_model.fit(X_train_text, train_labels)

test_texts = [
    "Чи варто зараз купувати євро та відкривати рахунок?",
    "Мій сервер на Ubuntu перестав відповідати на запити бази",
    "Отримання кредитної картки з великим кредитним лімітом",
    "Написання алгоритму сортування масиву великих даних"
]
test_labels_true = ["Finance", "IT", "Finance", "IT"]

X_test_text = vectorizer.transform(test_texts)
y_pred_text = nb_model.predict(X_test_text)

print("Прогнози:")
for text, pred in zip(test_texts, y_pred_text):
    print(f" - '{text}' => Клас '{pred}'")

print("\nОцінка ефективності текстової класифікації:")
print(f"Точність (Accuracy): {accuracy_score(test_labels_true, y_pred_text):.2f}")
print(f"Чутливість (Recall для IT): {recall_score(test_labels_true, y_pred_text, pos_label='IT'):.2f}")
print(f"Точність передбачення (Precision для IT): {precision_score(test_labels_true, y_pred_text, pos_label='IT'):.2f}")

# ЕТАП 6: Графова мережа
print("\nЕтап 6: Виявлення спільнот у мережі")

G = nx.karate_club_graph()

# Знаходимо спільноти
communities = list(community.greedy_modularity_communities(G))
print(f"У мережі виявлено {len(communities)} кластери (спільноти).")

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
colors = ['red', 'blue', 'green']

for i, comm in enumerate(communities):
    nx.draw_networkx_nodes(G, pos, nodelist=list(comm), node_color=colors[i], label=f'Спільнота {i+1}')
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Асоціативна кластеризація: Мережа зв'язків")
plt.legend()

graph_path = os.path.join(BASE_DIR, 'graph_plot.png')
plt.savefig(graph_path)
print(f"Графік мережі збережено як '{graph_path}'")