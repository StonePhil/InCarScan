import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classify","detect"], required=True, help="Тип задачи")
    ap.add_argument("--model", required=True, help="Путь к весам модели (.pt)")
    ap.add_argument("--source", required=True, help="Путь к картинке/папке/видео")
    ap.add_argument("--imgsz", type=int, default=None, help="Размер входа (опционально)")
    ap.add_argument("--out", default="runs/predict", help="Куда складывать результаты")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.predict(source=args.source, imgsz=args.imgsz, save=True, project=args.out, name=args.task)
    # Ultralytics сам сохранит картинки/видео с боксами/масками и JSON с предсказаниями.
    print(f"Saved results to: {os.path.join(args.out, args.task)}")

if __name__ == "__main__":
    main()