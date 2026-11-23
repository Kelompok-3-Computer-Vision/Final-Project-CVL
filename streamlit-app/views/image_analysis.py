"""Image Analysis page."""

import time

import cv2
import numpy as np
import streamlit as st

from model_utils import detect_largest_face, predict_drowsiness_timed
from overlay import annotate_frame
from theme import kv_rows, page_header, prediction_card, show_image


def _reset():
    st.session_state["img_upload_n"] = st.session_state.get("img_upload_n", 0) + 1


def render(ctx):
    models, cascade = ctx["models"], ctx["cascade"]
    page_header(
        "Image Analysis",
        "Upload a photo of a driver's face to classify it as drowsy or not drowsy.",
    )

    n = st.session_state.get("img_upload_n", 0)
    uploaded = st.file_uploader(
        "Drag and drop an image here, or browse",
        type=["jpg", "jpeg", "png"],
        key=f"img_upload_{n}",
        help="Supported formats: JPG, JPEG, PNG.",
    )

    if uploaded is None:
        st.markdown(
            '<div class="dds-note">The largest face in the image is detected automatically and '
            "cropped before classification.</div>",
            unsafe_allow_html=True,
        )
        return

    frame = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
        return

    t0 = time.perf_counter()
    face_box = detect_largest_face(frame, cascade)
    detect_ms = (time.perf_counter() - t0) * 1000.0

    left, right = st.columns([1.7, 1], gap="large")

    if face_box is None:
        with left:
            show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=uploaded.name)
        with right:
            st.markdown(
                prediction_card("No face detected", "neutral", "No face"), unsafe_allow_html=True
            )
            st.markdown(
                '<div class="dds-note" style="margin-top:.7rem">Try a clearer, front-facing photo '
                "where the face is at least 80×80 pixels.</div>",
                unsafe_allow_html=True,
            )
            st.button("Analyze another image", on_click=_reset, key="img_reset_nf")
        return

    x, y, w, h = face_box
    result = predict_drowsiness_timed(frame[y:y + h, x:x + w], models)
    annotated, is_drowsy = annotate_frame(frame.copy(), face_box, result)

    with left:
        show_image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=uploaded.name)

    with right:
        tone = "alert" if is_drowsy else "ok"
        st.markdown(
            prediction_card(
                result["label"], tone, "Drowsiness detected" if is_drowsy else "Normal",
                result["prob_drowsy"], result["confidence"],
            ),
            unsafe_allow_html=True,
        )
        t = result["timings_ms"]
        st.markdown(
            '<div class="dds-card" style="margin-top:.9rem"><div class="dds-label" style="margin-bottom:.4rem">'
            "Class probabilities</div>"
            + kv_rows([(k, f"{v * 100:.1f}%") for k, v in result["probs"].items()])
            + "</div>"
            '<div class="dds-card" style="margin-top:.9rem"><div class="dds-label" style="margin-bottom:.4rem">'
            "Processing time</div>"
            + kv_rows(
                [
                    ("Face detection", f"{detect_ms:.0f} ms"),
                    ("Preprocessing", f"{t['preprocess']:.1f} ms"),
                    ("MobileNetV2", f"{t['features']:.0f} ms"),
                    ("XGBoost", f"{t['xgboost']:.1f} ms"),
                ]
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)
        st.button("Analyze another image", on_click=_reset, key="img_reset")
