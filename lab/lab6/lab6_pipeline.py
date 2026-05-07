import os
import pandas as pd

print("=== ЗАПУСК КОНВЕЄРА ДАНИХ (ETL) ===\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("1. Extract: Завантаження сирих даних...")
csv_path = os.path.join(BASE_DIR, 'credit_risk_dataset.csv')

try:
    raw_data = pd.read_csv(csv_path)
    print(f"   Завантажено рядків: {len(raw_data)}")
except FileNotFoundError:
    print(f"ПОМИЛКА: Файл '{csv_path}' не знайдено!")
    exit()

print("2. Transform: Виконання операцій перетворення...")

transformed_data = raw_data.dropna().copy()

transformed_data = transformed_data[
    (transformed_data['person_age'] < 100) & 
    (transformed_data['person_emp_length'] < 60)
]

transformed_data['Income_Category'] = pd.cut(
    transformed_data['person_income'],
    bins=[0, 30000, 70000, float('inf')],
    labels=['Low', 'Medium', 'High']
)

# Обчислюємо співвідношення кредиту до доходу у відсотках
transformed_data['Credit_Burden_Percent'] = (transformed_data['loan_amnt'] / transformed_data['person_income'] * 100).round(2)

# Групуємо за метою кредиту і рахуємо середню суму та середню ставку
aggregated_summary = transformed_data.groupby('loan_intent').agg({
    'loan_amnt': 'mean',
    'loan_int_rate': 'mean'
}).round(2).rename(columns={'loan_amnt': 'Avg_Loan_Amount', 'loan_int_rate': 'Avg_Interest_Rate'})

print(f"   Трансформацію завершено. Рядків після очищення: {len(transformed_data)}")

print("3. Load: Збереження результатів...")

cleaned_path = os.path.join(BASE_DIR, 'cleaned_credit_data.csv')
aggregated_path = os.path.join(BASE_DIR, 'aggregated_loan_summary.csv')

transformed_data.to_csv(cleaned_path, index=False)

aggregated_summary.to_csv(aggregated_path)

print("\nКОНВЕЄР УСПІШНО ЗАВЕРШИВ РОБОТУ")
print("\nРЕЗУЛЬТАТ АГРЕГАЦІЇ")
print(aggregated_summary)
