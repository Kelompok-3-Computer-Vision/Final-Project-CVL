"""Overview page."""

import streamlit as st

from nav import go
from theme import page_header

MODES = [
    ("Live Webcam", "Continuous detection from your camera, with live FPS and inference latency."),
    ("Image Analysis", "Classify a single photo of a driver's face."),
    ("Video Analysis", "Analyze a recorded driving video and get a per-frame summary."),
]


def render(ctx):
    models = ctx["models"]
    h, w = models["image_size"]
    labels = " / ".join(models["class_indices"])

    page_header(
        "Drowsiness Detection System",
        "Classifies a driver's face as drowsy or not drowsy from a camera feed, an image or a video.",
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("## How it works")
        st.markdown(
            "Driver drowsiness slows reactions and attention. Visible cues in the face, such as "
            "drooping eyelids, are one signal a vision system can learn to recognise. "
            f"This application detects the driver's face in each frame and classifies it as **{labels}**."
        )
        st.markdown(
            "Each result is a per-frame classification. It is a decision-support demonstration "
            "and not a certified safety system."
        )

        st.markdown("## Analysis modes")
        for name, desc in MODES:
            c1, c2 = st.columns([4, 1], gap="medium")
            c1.markdown(
                f'<div class="dds-row"><span class="h">{name}</span><span class="p">{desc}</span></div>',
                unsafe_allow_html=True,
            )
            c2.button("Open", key=f"ov_go_{name}", on_click=go, args=(name,), use_container_width=True)

    with right:
        st.markdown("## Model architecture")
        steps = [
            ("Face detection", "OpenCV Haar cascade finds the largest frontal face in the frame."),
            ("Preprocessing", f"The face crop is converted to RGB, resized to {w}×{h} and scaled to the 0–1 range."),
            ("MobileNetV2", "A frozen convolutional network extracts a feature vector from the face."),
            ("XGBoost", "A gradient-boosted tree classifier maps the features to class probabilities."),
            ("Prediction", f"The most probable class ({labels}) and its confidence are reported."),
        ]
        body = "".join(
            f'<div class="dds-step"><div class="n">{i}</div><div><div class="h">{t}</div>'
            f'<div class="p">{d}</div></div></div>'
            for i, (t, d) in enumerate(steps, 1)
        )
        st.markdown(f'<div class="dds-card" style="padding:.4rem 1.25rem">{body}</div>', unsafe_allow_html=True)
