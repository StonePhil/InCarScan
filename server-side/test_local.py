# test_local.py
import os
from ai_server import analyze_file   # <-- Импорт функции из твоего ai_server.py (если там есть такая логика)

# путь к картинке на его ноуте
image_path = "uploads/car.jpg"

if not os.path.exists(image_path):
    print(f"Файл {image_path} не найден!")
else:
    # вызвать инференс
    result = analyze_file(image_path)
    print("Результат анализа:")
    print(result)
