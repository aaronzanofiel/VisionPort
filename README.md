# VisionPort 🚦

A vision-based traffic object detection system built using **YOLOv8n, PyTorch, Streamlit, and FastAPI**, with GPU acceleration through CUDA.

## Current Version

**v0.1 – Initial Prototype**

- Pretrained YOLOv8n object detection model.
- Streamlit-based application interface.
- FastAPI inference gateway.
- Request tracing support.
- CUDA-enabled PyTorch environment.

## Codebase Tour

| Component            | Location                    | Purpose                                             |
| -------------------- | --------------------------- | --------------------------------------------------- |
| Data Pipeline        | `data/`, `train/dataset.py` | Loading and preparing images and annotations        |
| Training Loop        | `train/train.py`            | Model training, loss calculation, and checkpointing |
| Architecture         | `train/model.py`            | YOLO-style backbone, neck, and detection head       |
| Evaluation Harness   | `eval/`                     | mAP, inference latency, and throughput measurements |
| Optimisation         | `optimise/`                 | ONNX export and TensorRT optimisation               |
| Model Serving        | `serve/`                    | FastAPI gateway and request tracing                 |
| Triton Configuration | `serve/triton/`             | Planned model repository and configuration          |

*Note: Some codebase components are planned for future development.*

## Project Structure

```text
VisionPort/
│
├── app.py
├── env_check.py
├── env_report.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── data/
├── models/
│   └── yolov8n.pt
│
├── train/
├── eval/
├── optimise/
│
├── serve/
│   ├── __init__.py
│   ├── gateway.py
│   └── gateway_trace.py
│
├── logs/
├── docs/
└── failures/
```

## Technology Stack

- **Programming Language:** Python
- **Object Detection:** YOLOv8n
- **Deep Learning:** PyTorch
- **GPU Acceleration:** NVIDIA CUDA
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **API Server:** Uvicorn
- **Version Control:** Git and GitHub

## Development Environment

| Component         | Configuration                        |
| ----------------- | ------------------------------------ |
| GPU               | NVIDIA GeForce RTX 3050 A Laptop GPU |
| PyTorch           | 2.14.1+cu130                         |
| CUDA Build        | 13.0                                 |
| CUDA Availability | Enabled                              |
| Python            | Project virtual environment          |

For detailed environment information, refer to `env_report.md`.

### Installation

Clone the repository:

```bash
git clone https://github.com/sandeep-dev07/VisionPort.git
cd VisionPort
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## How to Run

### 1. Start the FastAPI Gateway

```bash
python -m uvicorn serve.gateway_trace:app --reload
```

Open the API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Start the Streamlit Application

Open another terminal in the project root:

```bash
streamlit run app.py
```

## Known Limitations

- Currently uses pretrained YOLOv8n weights.
- Not specifically trained on Indian traffic datasets.
- Custom model training and evaluation pipelines are planned.
- ONNX and TensorRT optimisation are planned future enhancements.
- NVIDIA GPU acceleration depends on compatible hardware and drivers.

##
