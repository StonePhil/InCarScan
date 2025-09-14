# InCarScan

Демо‑приложение для загрузки трёх фотографий автомобиля (спереди/сбоку/сзади), их анализа моделью YOLOv8 и отображения результатов (включая аннотированные изображения). Проект состоит из двух частей:
- **Клиент**: Flask‑приложение, отдаёт страницы (`index`, `upload`, `results`, `report`), принимает файлы и проксирует их на ИИ‑сервер.
- **ИИ‑сервер**: Flask + Ultralytics YOLO, принимает три изображения, выполняет инференс, сохраняет аннотированные кадры и возвращает JSON с детекциями и ссылками на изображения.

---

## Архитектура

```
[Браузер] → [Клиент Flask /analyze] → (LAN) → [ИИ‑сервер Flask /analyze (YOLO)]
                                     ← JSON ←
[Клиент] сохраняет JSON в sessionStorage и редиректит на /results, где показывает результат
```

---

## Структура репозитория

```
InCarScan/
├─ app.py                      # Клиентский Flask (страницы + прокси /analyze)
├─ templates/
│  ├─ index.html
│  ├─ upload.html
│  ├─ results.html
│  └─ report.html
├─ static/
│  └─ car2/
│     ├─ unnamed.jpg          # логотип
│     ├─ user.png             # аватар в шапке
│     └─ loading.gif          # анимация на странице результатов
├─ uploads/                   # создаётся автоматически (на клиенте) по мере надобности
└─ README.md
```

ИИ‑сервер запускается на отдельной машине/ноутбуке и хранит свои веса по пути:
```
runs/detect_custom/weights/best.pt
```
Аннотированные изображения сохраняются в `uploads/annotated/` на ИИ‑сервере и отдаются по маршруту `/annotated/<filename>`.

---

## Зависимости

Клиент:
```bash
python -m pip install flask requests
```

ИИ‑сервер:
```bash
python -m pip install flask ultralytics opencv-python
```

Python 3.9+ (подходит и 3.13).

---

## Конфигурация IP адреса

В `app.py` (клиент) укажите IP и порт ИИ‑сервера в переменной `ai_url`:
```python
ai_url = "http://192.168.10.3:5000/analyze"  # IP:порт машины с ИИ
```
ИИ‑сервер должен слушать `0.0.0.0:5000`, чтобы быть доступным по LAN.

---

## Запуск

### 1) Клиент (ваш ноутбук)
```bash
python app.py
```
Откройте в браузере: `http://127.0.0.1:5000/`

Маршруты:
- `/` — главная страница
- `/upload` — загрузка трёх изображений (предпросмотр + удаление)
- `/results` — страница результатов (гиф → Success → вывод детекций и изображений)
- `/report` — заглушка страницы отчёта
- `/analyze` — POST‑эндпоинт для приёма трёх файлов и пересылки их на ИИ‑сервер

### 2) ИИ‑сервер (ноутбук с моделью YOLOv8)

Сохраните **следующий код** как `ai_server.py` и запустите:

```python
from flask import Flask, request, jsonify, send_from_directory
import os
from ultralytics import YOLO

app = Flask(__name__)

# === Загружаем модель один раз при старте ===
MODEL_PATH = os.path.join("runs", "detect_custom", "weights", "best.pt")
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
ANNOTATED_FOLDER = os.path.join(UPLOAD_FOLDER, "annotated")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Функция для инференса
def run_inference(image_path: str, save_name: str):
    """
    Прогоняет картинку через YOLOv8 и сохраняет аннотированное изображение.
    Возвращает список детекций и путь к аннотированному файлу.
    """
    results = model.predict(image_path, save=False)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results[0].names[cls_id]
        detections.append({
            "class": label,
            "confidence": round(conf, 3)
        })

    # Сохраняем картинку с боксами и лейблами
    annotated = results[0].plot()  # numpy array с разметкой
    annotated_path = os.path.join(ANNOTATED_FOLDER, save_name)
    import cv2
    cv2.imwrite(annotated_path, annotated)

    return detections, annotated_path


@app.route("/analyze", methods=["POST"])
def analyze():
    required = ["photo1", "photo2", "photo3"]
    for key in required:
        if key not in request.files:
            return jsonify({"error": f"Missing file {key}"}), 400

    results = {}

    for key in required:
        f = request.files[key]
        filename = f.filename
        if not os.path.splitext(filename)[1]:
            filename += ".jpg"

        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)

        detections, annotated_path = run_inference(save_path, filename)

        # Добавляем в результат список объектов + ссылку на картинку
        results[key] = {
            "detections": detections,
            "annotated_url": f"/annotated/{filename}"
        }

    return jsonify({"status": "ok", "results": results})


# Роут для отдачи аннотированных изображений
@app.route("/annotated/<path:filename>")
def serve_annotated(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

Запуск:
```bash
python ai_server.py
```
Проверьте, что эндпоинт доступен по адресу `http://<IP‑ИИ>:5000/analyze` (GET вернёт 405, это нормально).

---

## Код клиента (app.py)

Файл `app.py` уже содержит всё необходимое для маршрутов страниц и пересылки файлов на ИИ‑сервер:

```python
import requests
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/results")
def results_page():
    return render_template("results.html")

@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    files = {}
    for key in ["photo1", "photo2", "photo3"]:
        file = request.files[key]
        # Use the original filename if it exists
        filename = file.filename if file.filename else f"{key}.jpg"
        files[key] = (filename, file.stream, file.mimetype)

    ai_url = "http://192.168.10.3:5000/analyze"  # friend's IP
    response = requests.post(ai_url, files=files)

    return response.json(), response.status_code

if __name__ == "__main__":
    app.run(debug=True)
```

> На фронтенде `upload.html` собирает `FormData` из трёх файлов, шлёт на `/analyze`, помещает ответ в `sessionStorage` и делает редирект на `/results`.

---

## Формат ответа ИИ‑сервера

ИИ‑сервер возвращает JSON:
```json
{
  "status": "ok",
  "results": {
    "photo1": {
      "detections": [
        { "class": "dent", "confidence": 0.91 }
      ],
      "annotated_url": "/annotated/photo1.jpg"
    },
    "photo2": {
      "detections": [],
      "annotated_url": "/annotated/photo2.jpg"
    },
    "photo3": {
      "detections": [
        { "class": "scratch", "confidence": 0.75 }
      ],
      "annotated_url": "/annotated/photo3.jpg"
    }
  }
}
```

Если `annotated_url` относительный, на странице результатов используйте префикс `AI_BASE = "http://<IP‑ИИ>:5000"` для `<img src="...">`.

---

## Тестирование

1. Откройте `http://127.0.0.1:5000/upload` и выберите три изображения (`jpg/jpeg/png/bmp`).  
2. Нажмите «Отправить на проверку».  
3. Клиент отправит файлы на `/analyze`, получит JSON, сохранит его и перейдёт на `/results`.  
4. На `/results` через ~5 секунд гифка скроется и появится “Success!” + детекции и аннотированные изображения (если включены на ИИ‑сервере).

---

## FAQ / Трюки

- **404 на /results** — переходите по маршруту `/results`, а не `results.html`; ссылки в шаблонах делайте через `url_for(...)`.
- **Нет детекций на странице** — проверьте DevTools → Application → Session Storage → ключ `analysisResult`. Убедитесь, что JSON кладётся в `sessionStorage` после запроса.
- **ИИ не получает файлы** — проверьте `ai_url` (IP/порт), доступность по сети и фаерволл на машине с ИИ. ИИ‑сервер должен слушать `0.0.0.0`.
- **Имя файла без расширения** — в прокси указываются `(filename, stream, mimetype)`; на ИИ‑сервере при сохранении, если расширения нет, добавляется `.jpg`.

---

## Лицензия

MIT.
