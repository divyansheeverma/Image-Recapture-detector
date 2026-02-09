import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


# Helper: non-blocking debug wait (AUTO CLOSE)
def _debug_wait(ms=1200):
    """
    Non-blocking wait:
      - shows windows for `ms` milliseconds
      - you can press 'q' or ESC to close early
    """
    t = 0
    step = 20
    while t < ms:
        k = cv2.waitKey(step) & 0xFF
        if k == ord('q') or k == 27:  # q or ESC
            break
        t += step
    cv2.destroyAllWindows()


# =========================
# A) EXIF CHECK
# =========================
def check_exif(image_path: str):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return False, "Warning: No EXIF data (common for screenshots/WhatsApp/compressed images)", {}

        details = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            details[tag_name] = value

        has_make = "Make" in details and str(details["Make"]).strip() != ""
        has_model = "Model" in details and str(details["Model"]).strip() != ""

        subset = {
            "Make": details.get("Make"),
            "Model": details.get("Model"),
            "DateTimeOriginal": details.get("DateTimeOriginal"),
            "Software": details.get("Software"),
        }

        if has_make and has_model:
            return True, "EXIF present (Camera Make/Model available)", subset

        return False, "Potential signal: EXIF present but missing Camera Make/Model", subset

    except Exception as e:
        return False, f"EXIF check failed: {e}", {}


# ============================================================
# B) RIPPLING / MOIRÉ / BANDING CHECK (RECATURE ARTIFACTS)
# ============================================================
def _fft_energy_profile(gray, crop_center_ratio=0.9):
    h, w = gray.shape[:2]
    ch, cw = int(h * crop_center_ratio), int(w * crop_center_ratio)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    g = gray[y0:y0 + ch, x0:x0 + cw]

    g = g.astype(np.float32)
    g -= np.mean(g)

    F = np.fft.fft2(g)
    F = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F))

    H, W = mag.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rmax = r.max()

    radial = np.zeros(rmax + 1, dtype=np.float32)
    counts = np.zeros(rmax + 1, dtype=np.int32)

    np.add.at(radial, r, mag)
    np.add.at(counts, r, 1)
    radial = radial / np.maximum(counts, 1)

    return mag, radial


def _peakiness_score(profile, low_frac=0.06, high_frac=0.35):
    n = len(profile)
    lo = int(n * low_frac)
    hi = int(n * high_frac)
    if hi <= lo + 5:
        return 0.0

    band = profile[lo:hi].astype(np.float32)
    band = band - np.min(band)
    mean = float(np.mean(band)) + 1e-6
    std = float(np.std(band))

    mx = float(np.max(band))
    score = (mx - mean) / (std + 1e-6)
    return float(1.0 - np.exp(-max(0.0, score) / 3.0))


def _banding_score(gray):
    g = gray.astype(np.float32)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    row_mean = np.mean(g, axis=1)
    col_mean = np.mean(g, axis=0)

    def one_d_score(sig):
        sig = sig - np.mean(sig)
        F = np.fft.rfft(sig)
        mag = np.log1p(np.abs(F))
        if len(mag) < 20:
            return 0.0
        band = mag[5:int(len(mag) * 0.5)]
        m = float(np.mean(band)) + 1e-6
        s = float(np.std(band)) + 1e-6
        peak = float(np.max(band))
        raw = (peak - m) / s
        return float(1.0 - np.exp(-max(0.0, raw) / 3.0))

    return float(max(one_d_score(row_mean), one_d_score(col_mean)))


def recapture_artifact_score(image_bgr, resize_max=900, debug=False):
    h, w = image_bgr.shape[:2]
    scale = 1.0
    img = image_bgr

    if max(h, w) > resize_max:
        scale = resize_max / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, radial = _fft_energy_profile(gray, crop_center_ratio=0.9)
    peak_score = _peakiness_score(radial)
    band_score = _banding_score(gray)

    score = 0.65 * peak_score + 0.35 * band_score
    score = float(np.clip(score, 0.0, 1.0))

    dbg = {}
    if debug:
        dbg = {"peak_score": peak_score, "band_score": band_score, "scale_used": scale}
    return score, dbg


# ============================================================
# C) GLARE / SPECULAR HIGHLIGHT CHECK
# ============================================================
def glare_score(image_bgr,
                resize_max=900,
                v_thr=240,
                s_thr=45,
                min_blob_area=150,
                debug=False):
    h0, w0 = image_bgr.shape[:2]
    scale = 1.0
    img = image_bgr

    if max(h0, w0) > resize_max:
        scale = resize_max / float(max(h0, w0))
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = ((v >= v_thr) & (s <= s_thr)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    areas = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_blob_area:
            cleaned[labels == i] = 255
            areas.append(area)

    glare_area = int(np.sum(cleaned > 0))
    glare_ratio = glare_area / float(H * W + 1e-6)
    largest_blob_ratio = (max(areas) / float(H * W + 1e-6)) if areas else 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad_abs = np.abs(grad)

    border = cv2.morphologyEx(
        cleaned, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    border_px = grad_abs[border > 0]
    edge_strength = float(np.mean(border_px)) if border_px.size else 0.0

    score = 0.0
    score += min(1.0, glare_ratio / 0.05) * 0.65
    score += min(1.0, largest_blob_ratio / 0.02) * 0.25
    score += min(1.0, edge_strength / 25.0) * 0.10
    score = float(np.clip(score, 0.0, 1.0))

    dbg = {}
    if debug:
        dbg = {
            "glare_ratio": glare_ratio,
            "largest_blob_ratio": largest_blob_ratio,
            "edge_strength": edge_strength,
            "scale_used": scale,
            "glare_mask_small": cleaned
        }

    return score, dbg


# ============================================================
# 1) PAGE DETECTION + WARP
# ============================================================
def _order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_warp(image, pts4):
    rect = _order_points(pts4)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0],
                    [maxW - 1, 0],
                    [maxW - 1, maxH - 1],
                    [0, maxH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def _detect_page_by_whiteness(image_bgr,
                             resize_h=700,
                             v_min=150,
                             s_max=85,
                             min_area_ratio=0.18):
    H0, W0 = image_bgr.shape[:2]
    ratio = H0 / float(resize_h)
    img = cv2.resize(image_bgr, (int(W0 / ratio), resize_h), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, white_mask

    img_area = img.shape[0] * img.shape[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * float(min_area_ratio):
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)

        page_quad_full = (box * ratio).astype(np.float32)
        return page_quad_full, white_mask

    return None, white_mask


# ============================================================
# 2) EMBEDDED PHOTO DETECTION + FALSE POSITIVE REJECTION
# ============================================================
def _intersection_area(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a, b):
    inter = _intersection_area(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter + 1e-6
    return inter / union


def _remove_inner_boxes(boxes, contain_thr=0.90):
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        b_area = b[2] * b[3]
        inner = False
        for k in kept:
            inter = _intersection_area(b, k)
            if inter / (b_area + 1e-6) >= contain_thr:
                inner = True
                break
        if not inner:
            kept.append(b)
    return kept


def _dedupe_boxes(boxes, iou_thr=0.35, contain_thr=0.90):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return _remove_inner_boxes(kept, contain_thr=contain_thr)


def _texture_rejects_signature_or_text(crop_bgr):
    if crop_bgr.size == 0:
        return True

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    sat_std = float(np.std(hsv[:, :, 1]))

    if edge_density > 0.20 and sat_mean < 55:
        return True
    if lap_var < 35 and sat_mean < 40:
        return True
    if sat_std < 12 and sat_mean < 50 and edge_density > 0.08:
        return True

    return False


def _detect_embedded_photos(warped_bgr,
                           min_area_ratio=0.02,
                           max_area_ratio=0.75,
                           aspect_min=0.55,
                           aspect_max=3.5,
                           border_margin_ratio=0.02,
                           morph_k=13):
    H, W = warped_bgr.shape[:2]

    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s = cv2.GaussianBlur(s, (5, 5), 0)
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mx = int(W * border_margin_ratio)
    my = int(H * border_margin_ratio)

    candidates = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area_ratio = (w * h) / float(H * W)
        aspect = w / float(h)

        if not (min_area_ratio < area_ratio < max_area_ratio):
            continue
        if not (aspect_min < aspect < aspect_max):
            continue
        if x <= mx or y <= my or (x + w) >= (W - mx) or (y + h) >= (H - my):
            continue

        candidates.append((x, y, w, h))

    candidates = _dedupe_boxes(candidates, iou_thr=0.35, contain_thr=0.90)

    filtered = []
    for (x, y, w, h) in candidates:
        crop = warped_bgr[y:y + h, x:x + w]
        if _texture_rejects_signature_or_text(crop):
            continue
        filtered.append((x, y, w, h))

    filtered.sort(key=lambda b: (b[1], b[0]))
    return filtered, mask


# ============================================================
# MAIN
# ============================================================
def is_image_of_image_plus(image_path: str,
                           debug_show: bool = False,
                           resize_h_page: int = 700,
                           v_min: int = 150,
                           s_max: int = 85,
                           page_min_area_ratio: float = 0.18,
                           recapture_thr: float = 0.55,
                           glare_thr: float = 0.60,
                           use_recapture_fallback: bool = True,
                           debug_ms: int = 1200):
    info = {
        "page_found": False,
        "embedded_count": 0,
        "bboxes": [],
        "exif_ok": False,
        "exif_msg": "",
        "exif_details": {},
        "recapture_score": 0.0,
        "recapture_dbg": {},
        "glare_score": 0.0,
        "glare_dbg": {},
        "decision_reason": "",
        # ✅ NEW: images for Streamlit (BGR or single-channel masks)
        "page_contour_vis": None,
        "warped_with_boxes": None,
        "embedded_mask_vis": None,
        "white_mask_vis": None,
    }

    image = cv2.imread(image_path)
    if image is None:
        info["error"] = "Path incorrect or cannot read image."
        return False, info

    exif_ok, exif_msg, exif_details = check_exif(image_path)
    info["exif_ok"] = exif_ok
    info["exif_msg"] = exif_msg
    info["exif_details"] = exif_details

    rec_score, rec_dbg = recapture_artifact_score(image, debug=False)
    info["recapture_score"] = rec_score
    info["recapture_dbg"] = rec_dbg

    g_score, g_dbg = glare_score(image, debug=True if debug_show else True)  # keep glare mask always for UI
    info["glare_score"] = g_score
    info["glare_dbg"] = g_dbg

    page_quad, white_mask_dbg = _detect_page_by_whiteness(
        image, resize_h=resize_h_page, v_min=v_min, s_max=s_max, min_area_ratio=page_min_area_ratio
    )

    # keep page mask for UI
    info["white_mask_vis"] = white_mask_dbg

    if page_quad is None:
        if use_recapture_fallback:
            if (rec_score >= recapture_thr) or (g_score >= glare_thr):
                info["decision_reason"] = f"No page found; recapture={rec_score:.2f}, glare={g_score:.2f} triggered"
                return True, info

            if ((rec_score >= (recapture_thr - 0.12)) or (g_score >= (glare_thr - 0.15))) and (not exif_ok):
                info["decision_reason"] = f"No page found; recapture={rec_score:.2f}, glare={g_score:.2f}, EXIF weak/missing"
                return True, info

        info["decision_reason"] = "No page found; recapture/EXIF/glare signals not strong enough"
        return False, info

    info["page_found"] = True

    # ✅ NEW: draw page contour on original
    try:
        page_vis = image.copy()
        pts = page_quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(page_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
        info["page_contour_vis"] = page_vis
    except Exception:
        pass

    warped = _four_point_warp(image, page_quad)
    bboxes, photo_mask_dbg = _detect_embedded_photos(warped)

    info["embedded_mask_vis"] = photo_mask_dbg
    info["bboxes"] = bboxes
    info["embedded_count"] = len(bboxes)

    # ✅ NEW: warped with boxes
    try:
        warped_vis = warped.copy()
        for i, (x, y, w, h) in enumerate(bboxes, 1):
            cv2.rectangle(warped_vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                warped_vis, f"Photo {i}", (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        info["warped_with_boxes"] = warped_vis
    except Exception:
        pass

    result = len(bboxes) > 0
    info["decision_reason"] = "Page found + embedded photos detected" if result else "Page found but no embedded photos"

    if debug_show:
        show = warped.copy()
        for i, (x, y, w, h) in enumerate(bboxes, 1):
            cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(show, f"Photo {i}", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Warped Page (Final)", show)
        cv2.imshow("Saturation Mask (Embedded)", photo_mask_dbg)
        cv2.imshow("White Mask (Page)", white_mask_dbg)
        if "glare_mask_small" in info["glare_dbg"]:
            cv2.imshow("Glare Mask", info["glare_dbg"]["glare_mask_small"])
        _debug_wait(debug_ms)

    return result, info
