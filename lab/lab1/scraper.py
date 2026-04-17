import requests
from bs4 import BeautifulSoup
import csv
 
# Надсилаємо GET-запит для отримання вмісту сайту
url = 'https://www.example.com'
response = requests.get(url, verify=False)

# Аналізуємо HTML-вміст за допомогою Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

# Знаходимо перший екземпляр тегу <title> та витягуємо з нього текст
title = soup.find('title').get_text()
print(f"Знайдено заголовок: {title}")

# Зберігаємо отримані дані у структурований формат CSV
with open('data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([title])

print("Дані успішно збережено у файл data.csv")