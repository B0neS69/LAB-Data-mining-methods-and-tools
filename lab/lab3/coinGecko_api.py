import requests
import csv

# 1. Налаштування параметрів доступу до CoinGecko API
BASE_URL = 'https://api.coingecko.com/api/v3/coins/markets'

# Параметри запиту: базова валюта USD, сортування за капіталізацією, перші 10 монет
params = {
    'vs_currency': 'usd',
    'order': 'market_cap_desc',
    'per_page': 10,
    'page': 1,
    'sparkline': False
}

print("Відправляємо запит до CoinGecko API...")
extracted_data = []

try:
    # 2. Виконання HTTP GET запиту
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        
        # 3. Обробка та витягування потрібних даних
        for coin in data:
            name = coin['name']
            symbol = coin['symbol'].upper()
            price = coin['current_price']
            market_cap = coin['market_cap']
            price_change_24h = coin['price_change_percentage_24h']
            
            extracted_data.append([name, symbol, price, market_cap, price_change_24h])
            print(f"Отримано дані: {name} ({symbol}) - ${price}")
            
        # 4. Збереження оброблених даних у CSV файл
        csv_filename = 'crypto_market_data.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Назва', 'Символ', 'Ціна (USD)', 'Ринкова капіталізація', 'Зміна за 24 год (%)'])
            writer.writerows(extracted_data)
            
        print(f"\nУсі дані успішно експортовано у файл {csv_filename}")
    else:
        print(f"Помилка запиту. Код: {response.status_code}")

except Exception as e:
    print(f"Сталася помилка під час виконання: {e}")