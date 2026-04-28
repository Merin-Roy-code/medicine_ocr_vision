# 💊 Medicine Strip OCR — Google Cloud Vision + FastAPI

A **production-ready** Python backend that extracts structured information from
medicine strip images using Google Cloud Vision OCR, OpenCV preprocessing, and
RapidFuzz fuzzy matching.

---

## 📦 Project Structure

```
medicine_ocr_vision/
├── main.py           # FastAPI app + pipeline orchestrator
├── preprocess.py     # OpenCV image enhancement pipeline
├── ocr.py            # Google Cloud Vision integration
├── extractor.py      # Text cleaning + regex field extraction
├── fuzzy_match.py    # RapidFuzz medicine name / manufacturer matching
├── medicine_db.py    # In-memory medicine & manufacturer database
├── config.py         # Centralised env-var configuration
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / copy the project

```bash
cd medicine_ocr_vision
```

### 2. Create & activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Google Cloud Vision credentials

1. Create a GCP project and enable the **Cloud Vision API**.
2. Create a **Service Account** and download the JSON key.
3. Set the environment variable:

```powershell
# PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\your\key.json"

# bash / zsh
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
```

Or copy `.env.example` to `.env` and fill in the value — `python-dotenv` will
pick it up automatically.

### 5. Run the server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at **http://localhost:8000**.
Interactive docs: **http://localhost:8000/docs**

---

## 📡 API Reference

### `POST /extract`

Upload a medicine strip image and receive structured JSON.

**Request:** `multipart/form-data`

| Field   | Type | Description                           |
|---------|------|---------------------------------------|
| `image` | File | JPEG / PNG / WEBP / BMP / TIFF (≤15 MB) |

**Response:** `application/json`

```json
{
  "medicine_name": "Norfloxacin 400",
  "confidence": 0.93,
  "batch": "NF1234",
  "expiry": "05/2026",
  "manufacturer": "Cipla Ltd",
  "raw_ocr_text": "NORFLAXEM 400MG …",
  "processing_time_ms": 1842.5
}
```

**cURL example:**

```bash
curl -X POST http://localhost:8000/extract \
  -F "image=@/path/to/strip.jpg"
```

**Python example:**

```python
import requests

with open("strip.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/extract",
        files={"image": ("strip.jpg", f, "image/jpeg")},
    )
print(response.json())
```

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0" }
```

---

## ⚙️ Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable                        | Default  | Description                                |
|---------------------------------|----------|--------------------------------------------|
| `GOOGLE_APPLICATION_CREDENTIALS`| *(required)* | Path to GCP service account key JSON   |
| `PREPROCESS_TARGET_SIZE`        | `1600`   | Upscale long edge to this size (px)        |
| `CLAHE_CLIP_LIMIT`              | `2.0`    | CLAHE contrast clip limit                  |
| `DENOISE_H`                     | `6`      | NL-Means denoising strength                |
| `FUZZY_MIN_SCORE`               | `70`     | Min fuzzy score (0–100) to accept a match  |
| `API_HOST`                      | `0.0.0.0`| Server bind host                           |
| `API_PORT`                      | `8000`   | Server bind port                           |

---

## 🔬 Pipeline Architecture

```
Image Upload (JPEG/PNG)
       │
       ▼
┌─────────────────────────────────┐
│  Stage 1 — OpenCV Preprocessing │
│  • Upscale (INTER_CUBIC)        │
│  • LAB colour space             │
│  • CLAHE on L channel           │
│  • fastNlMeansDenoisingColored  │
│  • Unsharp-mask sharpening      │
│  • Gaussian-divide glare norm   │
└──────────────┬──────────────────┘
               │ PNG bytes
               ▼
┌─────────────────────────────────┐
│  Stage 2 — Google Cloud Vision  │
│  DOCUMENT_TEXT_DETECTION        │
│  • Full text annotation         │
│  • Per-block text + bounding    │
│    box (handles rotated strips) │
└──────────────┬──────────────────┘
               │ full_text + blocks
               ▼
┌─────────────────────────────────┐
│  Stage 3 — Text Cleaning        │
│  • Uppercase normalisation      │
│  • OCR char-level fixes         │
│    (0→O, |→I, ,→., etc.)       │
│  • Keyword fixes                │
│    (EXD→EXP, BAT→BATCH, …)    │
│  • Junk character removal       │
│                                 │
│  Regex Extraction               │
│  • Batch No / LOT               │
│  • Expiry → normalised MM/YYYY  │
│  • Manufacturer                 │
│  • Medicine name candidates     │
└──────────────┬──────────────────┘
               │ RawExtraction
               ▼
┌─────────────────────────────────┐
│  Stage 4 — RapidFuzz Matching   │
│  • WRatio scorer for medicine   │
│  • token_set_ratio for MFR      │
│  • Corrects OCR typos:          │
│    "Paracetarnol"→"Paracetamol" │
│  • Returns confidence 0.0–1.0   │
└──────────────┬──────────────────┘
               │ ExtractionResponse
               ▼
         JSON Response
```

---

## 🎯 Accuracy Improvement Guide

### Problem 1: Foil / Metallic Reflections

Foil packaging creates specular highlights that overexpose small text regions.

**Mitigation (already in preprocess.py):**
- `normalize_lighting()` uses Gaussian-divide to subtract the low-frequency
  illumination component. This "flattens" local brightness while keeping text.

**Additional tips:**
- Photograph under diffuse (indirect) light rather than direct sunlight.
- Take 2–3 photos at different angles; use the one with least glare.
- In code: detect overexposed regions (`mean(L) > 220`) and apply stronger
  CLAHE clip only in those regions.

```python
# Example: adaptive CLAHE per region
import cv2, numpy as np

def adaptive_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l = np.mean(l)
    clip = 3.5 if mean_l > 180 else 2.0   # stronger for bright/glare images
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return cv2.merge([clahe.apply(l), a, b])
```

---

### Problem 2: Blister Packaging Interference

Raised pill blisters cast shadows and curve the printed label, causing text
distortion.

**Mitigation:**
- Use homographic perspective correction if the strip is noticeably skewed.
- Detect and mask blister bubbles (bright circular regions) before OCR.

```python
# Example: blister bubble masking
import cv2, numpy as np

def mask_blisters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Find circular contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in contours:
        area = cv2.contourArea(c)
        if 500 < area < 50000:   # typical blister size range
            cv2.drawContours(mask, [c], -1, 255, -1)
    # Inpaint masked regions so OCR does not see blister reflections
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
```

---

### Problem 3: Small Fonts (Batch / Expiry)

Batch numbers and expiry dates are often ≤8pt, which OCR engines struggle with.

**Mitigation:**
- `upscale_if_small()` guarantees the long edge is ≥1600px — raising effective
  DPI to ~200–300 for typical strips.
- For even smaller text, use a **super-resolution** model:

```python
# Using OpenCV's DNN Super Resolution module (EDSR x4)
from cv2 import dnn_superres

sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")   # download from OpenCV model zoo
sr.setModel("edsr", 4)

img_4x = sr.upsample(img)    # 4× super resolution
```

---

## 🔀 Hybrid Approach: PaddleOCR + Google Cloud Vision

For maximum accuracy, use **two OCR engines** and merge their results.

### Why PaddleOCR?
| Feature | Google Vision | PaddleOCR |
|---------|--------------|-----------|
| Setup | Cloud API key | Local model |
| Curved text | Good | Excellent |
| Cost | Per-request | Free (local) |
| Offline | ❌ | ✅ |
| Languages | 100+ | 80+ |
| Small text | Very good | Good |

### Integration pattern

```python
# hybrid_ocr.py
from paddleocr import PaddleOCR
from ocr import run_ocr as vision_ocr  # Google Vision

_paddle = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

def hybrid_ocr(image_bytes: bytes) -> str:
    """
    Run both engines; merge text output.
    Prefer Vision for full-text; use PaddleOCR for curved/rotated sections.
    """
    import cv2, numpy as np
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Google Vision (primary — better for dense document text)
    vision_text, vision_blocks = vision_ocr(image_bytes)

    # PaddleOCR (secondary — better for curved blister labels)
    paddle_result = _paddle.ocr(img, cls=True)
    paddle_lines = [
        line[1][0]                          # text
        for line in (paddle_result[0] or [])
        if line[1][1] > 0.6                 # confidence filter
    ]
    paddle_text = "\n".join(paddle_lines)

    # Merge: use Vision as base; append PaddleOCR lines not already present
    merged_lines = list(vision_text.splitlines())
    for line in paddle_lines:
        if line not in merged_lines:
            merged_lines.append(line)

    return "\n".join(merged_lines)
```

**Add to requirements.txt:**
```
paddleocr==2.7.3
paddlepaddle==2.6.1   # CPU version
```

---

## 🧪 Testing

```bash
# Quick test via cURL
curl -X POST http://localhost:8000/extract \
  -F "image=@samples/norfloxacin_strip.jpg"

# Python integration test
python -c "
import requests, json
r = requests.post('http://localhost:8000/extract',
    files={'image': open('samples/norfloxacin_strip.jpg','rb')})
print(json.dumps(r.json(), indent=2))
"
```

---

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim

# OpenCV system deps
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t medicine-ocr .
docker run -p 8000:8000 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
  -v /path/to/key.json:/app/key.json \
  medicine-ocr
```

---

## 📈 Extending the Medicine Database

Edit `medicine_db.py` or load from an external source at startup:

```python
# Load from CSV at startup
import csv
from medicine_db import MEDICINE_DATABASE

def load_medicine_db(csv_path: str) -> None:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("medicine_name", "").strip()
            if name and name not in MEDICINE_DATABASE:
                MEDICINE_DATABASE.append(name)
```

For large databases (>10 000 entries) use `process.extractOne` with a
pre-built **BK-tree** index via `rapidfuzz.process` — it scales to millions
of entries without performance issues.

---

## 📄 License

MIT License © 2024
