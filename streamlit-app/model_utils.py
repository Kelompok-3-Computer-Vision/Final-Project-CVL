"""
model_utils.py
Utility functions to load the "Usulan" model (MobileNetV2 feature extractor + XGBoost)
and run drowsiness prediction on a single face image (BGR, as read by OpenCV).

Inference logic lives here and is independent of the UI. The preprocessing
(BGR->RGB, INTER_AREA resize to metadata image_size, /255.0) and the
classification logic are unchanged from the original implementation.
"""

import json
import os
import threading
import time

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_EXTRACTOR_FILENAME = "mobilenetv2_feature_extractor.keras"
XGB_FILENAME = "xgb_drowsiness_model.json"
METADATA_FILENAME = "metadata.json"

# Haar Cascade bawaan OpenCV untuk deteksi wajah.
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Keras/XGBoost objects are shared by the Streamlit script thread (image/video
# analysis) and the webcam worker thread, so inference is serialised.
_INFERENCE_LOCK = threading.Lock()


@st.cache_resource(show_spinner="Loading model (MobileNetV2 + XGBoost)...")
def load_models(model_dir: str = MODEL_DIR):
    """Load feature extractor, XGBoost classifier, dan metadata sekali saja (di-cache)."""
    # Candidate search paths in case model_dir is nested or relocated
    candidates = [
        model_dir,
        os.path.dirname(model_dir),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]
    resolved_dir = model_dir
    for candidate in candidates:
        if all(
            os.path.exists(os.path.join(candidate, fn))
            for fn in (FEATURE_EXTRACTOR_FILENAME, XGB_FILENAME, METADATA_FILENAME)
        ):
            resolved_dir = candidate
            break

    feature_extractor_path = os.path.join(resolved_dir, FEATURE_EXTRACTOR_FILENAME)
    xgb_path = os.path.join(resolved_dir, XGB_FILENAME)
    meta_path = os.path.join(resolved_dir, METADATA_FILENAME)

    for p in (feature_extractor_path, xgb_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File model tidak ditemukan: {p}\n"
                f"Pastikan folder '{resolved_dir}' berisi: {FEATURE_EXTRACTOR_FILENAME}, "
                f"{XGB_FILENAME}, dan {METADATA_FILENAME} (hasil sel 'Menyimpan Model' di notebook)."
            )

    feature_extractor = load_model(feature_extractor_path, compile=False)

    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_path)
    # Runtime setting only (weights/trees untouched): a single-row predict_proba
    # gains nothing from a thread pool and pays its start-up cost on every call.
    xgb_model.set_params(n_jobs=1)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    class_indices = metadata["class_indices"]          # e.g. {'Drowsy': 0, 'Non Drowsy': 1}
    index_to_class = {v: k for k, v in class_indices.items()}
    image_size = tuple(metadata["image_size"])           # e.g. (160, 160)

    # Cari label mana yang merepresentasikan "kantuk" (drowsy), tidak hardcode index.
    drowsy_label = None
    for name in class_indices:
        if "drowsy" in name.lower() and "non" not in name.lower():
            drowsy_label = name
            break
    if drowsy_label is None:
        # fallback: asumsikan label pertama (index 0) adalah kelas positif
        drowsy_label = index_to_class[0]

    models = {
        "feature_extractor": feature_extractor,
        "xgb_model": xgb_model,
        "index_to_class": index_to_class,
        "class_indices": class_indices,
        "image_size": image_size,
        "drowsy_label": drowsy_label,
    }

    # Warm-up: the first Keras call traces the graph and is much slower than the
    # following ones. Pay that cost here (once, at load) instead of on the first
    # webcam frame. A blank input is used; the result is discarded.
    _warm_up(models)
    return models


def _warm_up(models):
    try:
        h, w = models["image_size"]
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        predict_drowsiness_timed(blank, models)
    except Exception:  # warm-up must never prevent the app from starting
        pass


@st.cache_resource(show_spinner=False)
def load_face_cascade():
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("Gagal memuat Haar Cascade untuk deteksi wajah.")
    return cascade


def detect_largest_face(frame_bgr, cascade, min_size=(80, 80)):
    """Deteksi wajah terbesar pada frame. Return (x, y, w, h) atau None jika tidak ada wajah."""
    from settings import HAAR_MIN_NEIGHBORS, HAAR_SCALE_FACTOR
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # scaleFactor lebih kecil (1.05 vs 1.1) → lebih halus, bounding box lebih stabil antar frame
    # minNeighbors lebih besar (6 vs 5) → lebih sedikit false positive / bounding box jumping
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    # Ambil wajah dengan area terbesar (asumsi: pengemudi paling dekat ke kamera)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(w), int(h))


def preprocess_face(face_bgr, image_size):
    """Resize + rescale wajah (BGR) agar sesuai input MobileNetV2 (RGB, 0-1)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, image_size, interpolation=cv2.INTER_AREA)
    face_norm = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_norm, axis=0)  # (1, H, W, 3)


def _extract_features(feature_extractor, x):
    # predict_on_batch runs the same compiled inference graph as predict() (identical
    # output, verified bit-for-bit) but skips predict()'s per-call data-adapter and
    # callback setup, which dominated the cost of a single-image call (~60 ms vs ~9 ms
    # in our measurement). Calling the model directly (model(x)) was slower still.
    return np.asarray(feature_extractor.predict_on_batch(x))


def predict_drowsiness_timed(face_bgr, models):
    """
    Jalankan satu wajah (crop, BGR) melalui MobileNetV2 -> XGBoost.
    Return dict: {label, is_drowsy, prob_drowsy, confidence, probs, timings_ms}
    timings_ms holds measured wall-clock times: preprocess, features, xgboost, total.
    """
    t0 = time.perf_counter()
    x = preprocess_face(face_bgr, models["image_size"])
    pre_ms = _ms(t0, time.perf_counter())

    with _INFERENCE_LOCK:
        t1 = time.perf_counter()  # timing starts after the lock is held
        features = _extract_features(models["feature_extractor"], x)
        t2 = time.perf_counter()
        probs = models["xgb_model"].predict_proba(features)[0]  # urutan sesuai class_indices
        t3 = time.perf_counter()
    feat_ms = _ms(t1, t2)
    xgb_ms = _ms(t2, t3)

    index_to_class = models["index_to_class"]
    drowsy_label = models["drowsy_label"]
    drowsy_index = models["class_indices"][drowsy_label]
    prob_drowsy = float(probs[drowsy_index])

    pred_index = int(np.argmax(probs))
    pred_label = index_to_class[pred_index]

    return {
        "label": pred_label,
        "is_drowsy": pred_label == drowsy_label,
        "prob_drowsy": prob_drowsy,
        "confidence": float(probs[pred_index]),
        "probs": {index_to_class[i]: float(p) for i, p in enumerate(probs)},
        "timings_ms": {
            "preprocess": pre_ms,
            "features": feat_ms,
            "xgboost": xgb_ms,
            "total": pre_ms + feat_ms + xgb_ms,
        },
    }


def _ms(a, b):
    return (b - a) * 1000.0


def predict_drowsiness(face_bgr, models):
    """Backward-compatible wrapper (same keys as before, plus confidence/timings_ms)."""
    return predict_drowsiness_timed(face_bgr, models)
