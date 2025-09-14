import gradio as gr
from ultralytics import YOLO
import numpy as np, cv2
from PIL import Image

# Задайте пути к вашим моделям (обновите после обучения)
CLASSIFY_MODEL_PATH = "runs/classify/weights/best.pt"
DETECT_MODEL_PATH   = "runs/detect/weights/best.pt"

try:
    cls_model = YOLO(CLASSIFY_MODEL_PATH)
except Exception:
    cls_model = None

try:
    det_model = YOLO(DETECT_MODEL_PATH)
except Exception:
    det_model = None

def classify_image(img: Image.Image):
    if cls_model is None:
        return {"error": "Классификационная модель не найдена. Обучите ее и обновите путь."}
    res = cls_model.predict(img, imgsz=224)[0]
    if res.probs is None:
        return {"error": "Нет вероятностей. Проверьте модель/ввод."}
    names = res.names
    top1 = int(res.probs.top1)
    top1conf = float(res.probs.top1conf)
    return {names[top1]: top1conf}

def detect_image(img: Image.Image):
    if det_model is None:
        return None, "Детекционная модель не найдена. Обучите ее и обновите путь."
    res = det_model.predict(img, imgsz=640)[0]
    # Нарисуем результат с помощью встроенного plot():
    plotted = res.plot()  # ndarray BGR
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(plotted), ""

with gr.Blocks(title="Torch CV Starter") as demo:
    gr.Markdown("# Torch CV Starter — Классификация и Детекция")

    with gr.Tab("Классификация (clean/dirty)"):
        cls_in = gr.Image(type="pil", label="Загрузите фото")
        cls_out = gr.Label(label="Предсказание")
        gr.Examples(
            examples=[],
            inputs=cls_in
        )
        cls_btn = gr.Button("Предсказать")
        cls_btn.click(fn=classify_image, inputs=cls_in, outputs=cls_out)

    with gr.Tab("Детекция (scratches/dents)"):
        det_in = gr.Image(type="pil", label="Загрузите фото")
        det_vis = gr.Image(type="pil", label="Результат")
        det_msg = gr.Textbox(label="Сообщение", interactive=False)
        det_btn = gr.Button("Обнаружить")
        det_btn.click(fn=detect_image, inputs=det_in, outputs=[det_vis, det_msg])

if __name__ == "__main__":
    demo.launch()