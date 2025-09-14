# Torch CV Starter (Ultralytics YOLO + Gradio)

Готовый минимальный проект на **PyTorch/Ultralytics** для двух задач:
- **Классификация** (clean/dirty).
- **Детекция** (scratches/dents).

## Установка
```bash
pip install -r requirements.txt
```

## Структура данных
### Классификация
```
data/classification/
  train/
    clean/...
    dirty/...
  val/
    clean/...
    dirty/...
```
YAML: `data/classification/dataset.yaml` — имена классов.

### Детекция (YOLO формат)
```
data/detection/
  images/
    train/ *.jpg|png
    val/   *.jpg|png
  labels/
    train/ *.txt  # x_center y_center width height class_id (нормированные 0..1)
    val/   *.txt
```
YAML: `data/detection/dataset.yaml` — пути и имена классов.

## Обучение
### Классификация
```bash
bash train_cls.sh
```
### Детекция
```bash
bash train_det.sh
```

## Инференс через CLI
```bash
# Классификация
yolo task=classify mode=predict model=runs/classify/train/weights/best.pt source=path/to/img_or_dir

# Детекция
yolo task=detect  mode=predict model=runs/detect/train/weights/best.pt   source=path/to/img_or_dir
```

## Инференс через Python
```bash
python infer.py --task classify --model runs/classify/train/weights/best.pt --source path/to/img.jpg
python infer.py --task detect   --model runs/detect/train/weights/best.pt   --source path/to/dir_or_img
```

## Gradio демо (2 вкладки)
```bash
python gradio_app.py
```
Откроется локальный веб-интерфейс: вкладка **Классификация** и **Детекция**.

## Экспорт модели
```bash
bash export.sh
```

## Примечания
- Следите за балансом классов, используйте аугментации (Ultralytics включает базовые).
- Метрики: для классификации — accuracy/F1; для детекции — mAP/IoU.
- Для презентации покажите примеры правильных/ошибочных предсказаний и UX-интеграцию.