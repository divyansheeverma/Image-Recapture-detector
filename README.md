# Photo-of-Photo (Recapture) Detection using OpenCV

Detects whether an input image is likely a **recaptured image** (i.e., a photo taken of another photo / screen / printed image).

This pipeline combines:

- **EXIF camera metadata checks**
- **Recapture artifact detection** (FFT peakiness + banding signals)
- **Glare/specular highlight detection**
- **Page detection + perspective warp**
- **Embedded-photo detection** using saturation segmentation + contour filtering

---

## Features

- Detects embedded photos inside a detected page/document
- Fallback detection using recapture + glare + EXIF signals when page is not found
- Debug visualization windows with auto-close

---

## Setup

```bash
git clone <your-repo-link>
cd phott-OF-photo-detector
pip install -r requirements.txt
```
