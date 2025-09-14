#!/usr/bin/env bash
set -e
yolo task=classify mode=train data=data/classification/dataset.yaml model=yolov8n-cls.pt epochs=20 imgsz=224 project=runs name=classify