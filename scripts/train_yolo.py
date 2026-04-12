#!/usr/bin/env python3
"""Train YOLOv8n on cassini/issna with unbuffered logging."""
import os
import sys
import time
from pathlib import Path

# Unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path("/Users/upadhyay/dev/ICES/object_detection_opencv_cpp")
os.chdir(ROOT)

from ultralytics import YOLO
import torch

print(f"[{time.strftime('%H:%M:%S')}] torch {torch.__version__}")
print(f"[{time.strftime('%H:%M:%S')}] mps available: {torch.backends.mps.is_available()}")
print(f"[{time.strftime('%H:%M:%S')}] cuda available: {torch.cuda.is_available()}")
print(f"[{time.strftime('%H:%M:%S')}] cpu count: {os.cpu_count()}")

# Use CPU explicitly and small image size for reasonable speed on this Mac
device = "cpu"  # MPS has known stability issues with ultralytics train loop

print(f"[{time.strftime('%H:%M:%S')}] loading yolov8n.pt")
model = YOLO("yolov8n.pt")
print(f"[{time.strftime('%H:%M:%S')}] starting train")

results = model.train(
    data=str(ROOT / "data/cassini_issna_yolo/data.yaml"),
    epochs=50,
    imgsz=320,       # small for CPU speed; upscale at inference time
    batch=8,
    patience=10,
    seed=42,
    project=str(ROOT / "training_runs/cassini_issna"),
    name="run1",
    exist_ok=True,
    device=device,
    workers=2,
    verbose=True,
    save=True,
    save_period=1,
    plots=True,
    val=True,
    cache=True,      # cache images in RAM to avoid disk thrash on CPU
)

print(f"[{time.strftime('%H:%M:%S')}] training complete")
print(f"results dir: {results.save_dir if hasattr(results, 'save_dir') else 'unknown'}")
