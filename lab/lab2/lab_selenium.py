from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv

# 1. Ініціалізація браузера
driver = webdriver.Chrome()

try:
    # 2. Перехід на цільову веб-сторінку (Telemart, категорія ноутбуки)
    url = 'https://telemart.ua/ua/laptops/'
    driver.get(url)
    print("Сторінка завантажується, очікуємо появи товарів...")

    # 3. Налаштування явного очікування
    wait = WebDriverWait(driver, 15)
    
    # Чекаємо, поки з'явиться хоча б одна картка товару
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.product-item')))

    # 4. Пошук усіх карток товарів на поточній сторінці
    product_cards = driver.find_elements(By.CSS_SELECTOR, '.product-item')
    print(f"Знайдено товарів на сторінці: {len(product_cards)}. Розпочинаємо збір...")

    extracted_data = []

    # Збираємо дані
    for card in product_cards:
        try:
            # Використовуємо CSS-селектори для пошуку назви та ціни всередині картки
            title = card.find_element(By.CSS_SELECTOR, '.product-item__title').text
            price = card.find_element(By.CSS_SELECTOR, '.product-cost').text
            
            extracted_data.append([title, price])
        except Exception:
            continue

    # 5. Збереження зібраної інформації у зручному форматі CSV
    csv_filename = 'telemart_laptops.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Назва моделі', 'Ціна'])
        writer.writerows(extracted_data)
        
    print(f"Скрапінг завершено! Дані успішно збережено у файл {csv_filename}")

except Exception as e:
    print(f"Під час виконання сталася помилка: {e}")
    print("Примітка: можливо сайт відхилив запит через систему антибот-захисту.")

finally:
    driver.quit()