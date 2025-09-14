#!/usr/bin/env bash
set -e
# Классификация
yolo mode=export model=runs/classify/weights/best.pt format=onnx
# Детекция
yolo mode=export model=runs/detect/weights/best.pt format=onnx
# При необходимости:
# yolo mode=export model=runs/detect/weights/best.pt format=tflite
# yolo mode=export model=runs/detect/weights/best.pt format=torchscript