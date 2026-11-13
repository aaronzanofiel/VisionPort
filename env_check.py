
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

rows = []

def log(key, value):
    rows.append((key, str(value)))
    print(f"{key:<30}{value}")

def pkg(name):
    try:
        return version(name)
    except Exception:
        return "NOT INSTALLED"

log("Date", datetime.now().strftime("%Y-%m-%d %H:%M"))
log("Operating system", platform.platform())
log("Python version", sys.version.split()[0])
log("Python interpreter", sys.executable)

for name in ["torch", "torchvision", "ultralytics", "numpy",
             "pillow", "fastapi", "uvicorn", "streamlit", "pandas"]:
    log(f"pkg: {name}", pkg(name))

if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version",
         "--format=csv,noheader"],
        capture_output=True,
        text=True
    )
    log("NVIDIA driver",
        out.stdout.strip().splitlines()[0] if out.stdout else "unreadable")
else:
    log("NVIDIA driver", "nvidia-smi not found (no NVIDIA GPU or driver)")

try:
    import torch

    log("PyTorch CUDA build", torch.version.cuda or "None (CPU-only build)")
    log("torch.cuda.is_available()", torch.cuda.is_available())

    if torch.backends.cudnn.is_available():
        log("cuDNN version", torch.backends.cudnn.version())
    else:
        log("cuDNN version", "not available")

    if torch.cuda.is_available():
        log("GPU count", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log(f"GPU {i}", f"{p.name}, {p.total_memory / 1e9:.1f} GB")

        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()

        log("GPU matmul test", f"PASS (result on {y.device})")
    else:
        log("GPU matmul test", "SKIPPED (running on CPU)")

except Exception as e:
    log("PyTorch", f"FAILED to import or run: {type(e).__name__}")

lines = [
    "# VisionPort environment report",
    "",
    "| Item | Value |",
    "|---|---|"
]

lines += [f"| {k} | {v} |" for k, v in rows]

Path("env_report.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("\nSaved: env_report.md")
