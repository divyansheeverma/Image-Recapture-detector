# 📸 Photo-of-Photo (Recapture) Detection

> Detect whether an image is a recaptured photo — a photo taken of another photo, screen, or printed document — using a multi-signal computer vision pipeline.

---

## 🧠 How It Works

This project combines **5 independent detection signals** to determine if an image is recaptured:

| Signal | Method | What It Catches |
|---|---|---|
| 📷 EXIF Metadata | Camera tag analysis | Missing/inconsistent camera data |
| 📡 FFT Peakiness | Frequency domain analysis | Moiré patterns & banding artifacts |
| ✨ Glare Detection | Specular highlight analysis | Screen glare & surface reflections |
| 📄 Page Detection | Contour + perspective warp | Printed document boundaries |
| 🖼️ Embedded Photo | Saturation segmentation | Photos-within-photos on a surface |

When a **page/document is detected**, the pipeline focuses on finding an embedded photo inside it. Otherwise, it falls back to a combined recapture + glare + EXIF scoring approach.

---

## ✨ Features

- ✅ Multi-signal fusion for robust detection
- ✅ Handles screen captures, printed photos, and framed images
- ✅ Perspective correction via page boundary warping
- ✅ Debug visualization with auto-close windows
- ✅ Works on standard image formats (JPG, PNG, etc.)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/divyansheeverma/Image-Recapture-detector.git
cd Image-Recapture-detector

# Install dependencies
pip install -r requirements.txt

# Run detection on an image
python detect.py --image path/to/your/image.jpg
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

- **OpenCV** — Image processing, contour detection, perspective transforms
- **NumPy** — FFT computation and signal analysis
- **Pillow / piexif** — EXIF metadata extraction

---

## 📁 Project Structure

```
Image-Recapture-detector/
├── detect.py              # Main detection pipeline
├── signals/
│   ├── exif_check.py      # EXIF metadata analysis
│   ├── fft_analysis.py    # Frequency domain peakiness detection
│   ├── glare_detect.py    # Specular highlight detection
│   ├── page_detect.py     # Page/document boundary detection
│   └── embedded_photo.py  # Saturation-based photo segmentation
├── utils/
│   └── visualizer.py      # Debug visualization helpers
├── requirements.txt
└── README.md
```

---

## 🔍 Detection Logic

```
Input Image
    │
    ├── EXIF Check ──────────────────────┐
    ├── FFT Peakiness Analysis ──────────┤
    ├── Glare / Specular Detection ──────┤──► Fusion Score ──► RECAPTURED / AUTHENTIC
    ├── Page Detected? ──► YES ──────────┤
    │       └── Embedded Photo Search ───┘
    └── (Fallback: combined signal score)
```

---

## 📊 Example Output

```
[INFO] EXIF: No camera metadata found → suspicious
[INFO] FFT peakiness score: 0.83 → high (banding detected)
[INFO] Glare regions: 3 found
[INFO] Page boundary detected → searching for embedded photo...
[INFO] Embedded photo region found (confidence: 0.91)

RESULT: ⚠️  RECAPTURED IMAGE DETECTED
```
---


---

*Built with OpenCV & Python · by [Divyanshee Verma](https://github.com/divyansheeverma)*
