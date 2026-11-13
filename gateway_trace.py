"""VisionPort Session 3 - inference request trace

Same YOLOv8n model as Session 2, but predict() is opened up so every stage
(bytes -> image -> letterbox -> tensor -> GPU -> raw output -> NMS -> JSON)
is run by hand and reported with its data format, shape, device and time.

Run:   uvicorn gateway_trace:app --reload
Docs:  http://127.0.0.1:8000/docs   ->  POST /trace
"""
import io
import time

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

try:  # newer ultralytics versions moved NMS into utils.nms
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # older versions keep it in utils.ops
    from ultralytics.utils.ops import non_max_suppression

app = FastAPI(title="VisionPort Gateway - Session 3 request trace")

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolov8n.pt")
net = yolo.model.to(device).eval()  # the raw PyTorch network inside YOLO
names = yolo.names                  # class id -> class name
IMGSZ = 640


def sync():
    """GPU work is asynchronous: wait for it so timings are honest."""
    if device == "cuda":
        torch.cuda.synchronize()


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/trace")
def trace(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.45):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Send a JPEG or PNG image")

    stages = []

    def add(name, where, fmt, shape, t0, note=""):
        stages.append({
            "stage": len(stages) + 1, "name": name, "where": where,
            "format": fmt, "shape": shape, "note": note,
            "ms": round((time.perf_counter() - t0) * 1000, 2),
        })

    t_all = time.perf_counter()

    # 1. Receive the bytes of the uploaded file
    t0 = time.perf_counter()
    data = file.file.read()
    add("receive upload", "CPU / gateway", "bytes", f"{len(data)} bytes", t0)

    # 2. Decode into an image
    t0 = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")
    arr = np.array(image)  # H x W x 3, uint8, RGB
    orig_hw = arr.shape[:2]
    add("decode image", "CPU / gateway", "numpy uint8 (H,W,C)", list(arr.shape), t0)

    # 3. Letterbox: resize keeping aspect ratio, pad to 640 x 640
    t0 = time.perf_counter()
    boxed = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False)(image=arr)
    add("letterbox resize", "CPU / gateway", "numpy uint8 (H,W,C)", list(boxed.shape), t0,
        "aspect ratio kept, grey padding added")

    # 4. Build the input tensor: HWC uint8 -> NCHW float32 in [0, 1]
    t0 = time.perf_counter()
    x = torch.from_numpy(boxed).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255.0
    add("to tensor", "CPU / gateway", "torch float32 (N,C,H,W)", list(x.shape), t0,
        "channels first, batch dimension added, scaled to 0-1")

    # 5. Host -> device transfer (CPU memory to GPU memory)
    t0 = time.perf_counter()
    x = x.to(device)
    sync()
    add("host to device", f"CPU -> {device}", "torch float32 (N,C,H,W)", list(x.shape), t0,
        "copy over PCIe" if device == "cuda" else "no copy (already on CPU)")

    # 6. Model forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        out = net(x)
    sync()
    raw = out[0] if isinstance(out, (list, tuple)) else out
    add("model forward", device, "torch float32 (N,4+classes,anchors)", list(raw.shape), t0,
        "4 box values + 80 class scores for each of the anchor positions")

    # 7. Non-maximum suppression: drop low-confidence and duplicate boxes
    t0 = time.perf_counter()
    dets = non_max_suppression(raw, conf, iou)[0]  # (n, 6): x1,y1,x2,y2,conf,cls
    sync()
    add("NMS", device, "torch float32 (n,6)", list(dets.shape), t0,
        "columns: x1, y1, x2, y2, confidence, class id")

    # 8. Map boxes from the 640x640 letterboxed frame back to the original image
    t0 = time.perf_counter()
    dets[:, :4] = ops.scale_boxes(x.shape[2:], dets[:, :4], orig_hw)
    add("rescale boxes", device, "torch float32 (n,6)", list(dets.shape), t0,
        "boxes now in original image pixels")

    # 9. Device -> host and JSON serialisation
    t0 = time.perf_counter()
    rows = dets.cpu().tolist()
    detections = [
        {"class": names[int(c)], "confidence": round(s, 3),
         "box_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)]}
        for x1, y1, x2, y2, s, c in rows
    ]
    add("serialise response", "CPU / gateway", "JSON", f"{len(detections)} detections", t0)

    sync()
    return {
        "device": device,
        "original_size_hw": list(orig_hw),
        "stages": stages,
        "total_ms": round((time.perf_counter() - t_all) * 1000, 2),
        "detections": detections,
    }