#!/usr/bin/env bash
set -e
yolo task=detect mode=train data=data/detection/dataset.yaml model=yolov8n.pt epochs=40 imgsz=640 project=runs name=detect