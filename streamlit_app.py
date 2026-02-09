import streamlit as st
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

# ✅ Correct import (your detector is inside src/)
from src.detector import is_image_of_image_plus


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def list_images(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def safe_sigmoid(x: np.ndarray):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


st.set_page_config(page_title="Photo-of-Photo Detector", layout="wide")
st.title("📸 Photo-of-Photo (Image Re-capture) Detection Demo")

with st.sidebar:
    st.header("Settings")
    resize_h_page = st.slider("Page resize height", 400, 1200, 700, 50)
    v_min = st.slider("Whiteness V_min", 80, 220, 150, 5)
    s_max = st.slider("Whiteness S_max", 20, 140, 85, 5)
    page_min_area_ratio = st.slider("Page min area ratio", 0.05, 0.50, 0.18, 0.01)

    recapture_thr = st.slider("Recapture threshold", 0.10, 0.95, 0.55, 0.01)
    glare_thr = st.slider("Glare threshold", 0.10, 0.95, 0.60, 0.01)

    use_recapture_fallback = st.checkbox("Fallback when page not found", True)

    st.divider()
    st.caption("Tip: keep debug_show OFF for Streamlit (no OpenCV popups).")


tab1, tab2 = st.tabs(["✅ Single Image Demo", "📊 Batch Evaluation"])


# =========================================================
# TAB 1: Single Image Demo
# =========================================================
with tab1:
    st.subheader("Upload one image and see decision + contour/warp visualizations")

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        key="single"
    )

    if uploaded:
        suffix = Path(uploaded.name).suffix.lower() if Path(uploaded.name).suffix else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            img_path = tmp.name

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            st.error("Could not read image.")
        else:
            colA, colB = st.columns([1.2, 1])
            with colA:
                st.image(bgr_to_rgb(img_bgr), use_container_width=True, caption="Input")

            with colB:
                if st.button("Run Detection 🚀", type="primary"):
                    with st.spinner("Running detection..."):
                        pred_bool, info = is_image_of_image_plus(
                            img_path,
                            debug_show=False,
                            resize_h_page=resize_h_page,
                            v_min=v_min,
                            s_max=s_max,
                            page_min_area_ratio=page_min_area_ratio,
                            recapture_thr=recapture_thr,
                            glare_thr=glare_thr,
                            use_recapture_fallback=use_recapture_fallback,
                            debug_ms=0,
                        )

                    st.markdown("### Result")
                    if pred_bool:
                        st.error("⚠️ IMAGE-OF-IMAGE (Spoof suspected)")
                    else:
                        st.success("✅ REAL IMAGE")

                    st.write("**Reason:**", info.get("decision_reason", ""))

                    rec_s = float(info.get("recapture_score", 0.0) or 0.0)
                    gla_s = float(info.get("glare_score", 0.0) or 0.0)

                    rec_margin = (rec_s - recapture_thr) / max(1e-6, (1.0 - recapture_thr))
                    gla_margin = (gla_s - glare_thr) / max(1e-6, (1.0 - glare_thr))
                    raw = max(rec_margin, gla_margin)
                    conf_i1 = float(safe_sigmoid(np.array([raw]))[0])

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Recapture score", f"{rec_s:.3f}")
                    m2.metric("Glare score", f"{gla_s:.3f}")
                    m3.metric("Confidence (IoI)", f"{conf_i1:.2f}")
                    m4.metric("Embedded photos", str(info.get("embedded_count", 0)))

                    st.write("**Page found:**", info.get("page_found", False))
                    st.write("**EXIF:**", info.get("exif_msg", ""))
                    st.write("**EXIF OK:**", info.get("exif_ok", False))

                    st.divider()
                    st.markdown("## Visual Debug Outputs")

                    page_vis = info.get("page_contour_vis", None)
                    if isinstance(page_vis, np.ndarray):
                        st.write("**Page contour (original image)**")
                        st.image(bgr_to_rgb(page_vis), use_container_width=True)

                    warped_vis = info.get("warped_with_boxes", None)
                    if isinstance(warped_vis, np.ndarray):
                        st.write("**Warped page + embedded photo boxes**")
                        st.image(bgr_to_rgb(warped_vis), use_container_width=True)

                    white_mask = info.get("white_mask_vis", None)
                    if isinstance(white_mask, np.ndarray):
                        st.write("**White mask (page detection)**")
                        st.image(white_mask, clamp=True, use_container_width=True)

                    embedded_mask = info.get("embedded_mask_vis", None)
                    if isinstance(embedded_mask, np.ndarray):
                        st.write("**Embedded mask (photo detection)**")
                        st.image(embedded_mask, clamp=True, use_container_width=True)

                    glare_dbg = info.get("glare_dbg", {}) or {}
                    glare_mask = glare_dbg.get("glare_mask_small", None)
                    if isinstance(glare_mask, np.ndarray):
                        st.write("**Glare mask**")
                        st.image(glare_mask, clamp=True, use_container_width=True)


# =========================================================
# TAB 2: Batch Evaluation
# =========================================================
with tab2:
    st.subheader("Evaluate on two folders (image-of-image vs real)")

    col1, col2 = st.columns(2)
    with col1:
        pos_folder_in = st.text_input("Folder path: image-of-image (label=1)", value="")
    with col2:
        neg_folder_in = st.text_input("Folder path: real images (label=0)", value="")

    run_batch = st.button("Run Batch Evaluation 📊", type="primary")

    if run_batch:
        if not pos_folder_in or not neg_folder_in:
            st.error("Please provide both folder paths.")
            st.stop()

        pos_folder = Path(pos_folder_in)
        neg_folder = Path(neg_folder_in)

        if (not pos_folder.exists()) or (not neg_folder.exists()):
            st.error("One or both folder paths do not exist.")
            st.stop()

        pos_files = list_images(pos_folder)
        neg_files = list_images(neg_folder)

        if not pos_files and not neg_files:
            st.error("No images found in either folder.")
            st.stop()

        st.write(f"**Positives (1):** {len(pos_files)} images")
        st.write(f"**Negatives (0):** {len(neg_files)} images")

        rows = []
        total = len(pos_files) + len(neg_files)
        prog = st.progress(0)
        count = 0

        def run_one(path: Path, y_true: int):
            pred_bool, info = is_image_of_image_plus(
                str(path),
                debug_show=False,
                resize_h_page=resize_h_page,
                v_min=v_min,
                s_max=s_max,
                page_min_area_ratio=page_min_area_ratio,
                recapture_thr=recapture_thr,
                glare_thr=glare_thr,
                use_recapture_fallback=use_recapture_fallback,
                debug_ms=0,
            )

            y_pred = 1 if pred_bool else 0
            rec_s = float(info.get("recapture_score", 0.0) or 0.0)
            gla_s = float(info.get("glare_score", 0.0) or 0.0)

            rec_margin = (rec_s - recapture_thr) / max(1e-6, (1.0 - recapture_thr))
            gla_margin = (gla_s - glare_thr) / max(1e-6, (1.0 - glare_thr))
            raw = max(rec_margin, gla_margin)
            score = float(safe_sigmoid(np.array([raw]))[0])

            rows.append({
                "path": str(path),
                "y_true": y_true,
                "y_pred": y_pred,
                "score": score,
                "recapture_score": rec_s,
                "glare_score": gla_s,
                "page_found": info.get("page_found"),
                "embedded_count": info.get("embedded_count"),
                "exif_ok": info.get("exif_ok"),
                "exif_msg": info.get("exif_msg"),
                "decision_reason": info.get("decision_reason"),
            })

        with st.spinner("Running batch detection..."):
            for p in pos_files:
                run_one(p, 1)
                count += 1
                prog.progress(count / total)

            for p in neg_files:
                run_one(p, 0)
                count += 1
                prog.progress(count / total)

        df = pd.DataFrame(rows)

        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        y_score = df["score"].to_numpy()

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = float((y_true == y_pred).mean())

        st.markdown("### Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("TN", str(tn))
        m3.metric("FP", str(fp))
        m4.metric("FN", str(fn))
        st.metric("TP", str(tp))

        st.write("**Confusion Matrix** (rows=true [0,1], cols=pred [0,1])")
        st.dataframe(pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))

        st.write("**Classification Report**")
        rep = classification_report(y_true, y_pred, target_names=["real(0)", "image-of-image(1)"], digits=4)
        st.code(rep)

        auc_val = roc_auc_score(y_true, y_score)
        auc_flip = roc_auc_score(y_true, 1 - y_score)
        st.write(f"**ROC-AUC:** {auc_val:.4f}  |  **ROC-AUC (flipped):** {auc_flip:.4f}")

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.legend(loc="lower right")
        st.pyplot(fig)

        st.markdown("### Per-image output")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )
