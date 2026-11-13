import io

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="VisionPort Gateway (Session 2 prototype)")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {
        "project": "VisionPort",
        "message": "VisionPort FastAPI Gateway is running!",
        "status": "success",
        "device": device,
        "available_endpoints": [
            "/health",
            "/detect",
            "/docs"
        ]
    }

@app.post("/detect")
def detect(image: UploadFile = File(...), conf: float = 0.35):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Send a JPEG or PNG image")

    try:
        image = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    r = model.predict(image, conf=conf, device=device, verbose=False)[0]

    detections = [
        {
            "class": r.names[int(c)],
            "confidence": round(float(s), 3),
            "box_xyxy": [round(float(v), 1) for v in b],
        }
        for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)
    ]

    return {
        "image_size": {"width": image.width, "height": image.height},
        "num_detections": len(detections),
        "detections": detections,
        "timing_ms": {k: round(v, 2) for k, v in r.speed.items()},
    }