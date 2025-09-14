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
