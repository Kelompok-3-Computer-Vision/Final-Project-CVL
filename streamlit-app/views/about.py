"""About page. Model facts are read from the loaded model/metadata, not hard-coded."""

import streamlit as st

from theme import kv_rows, page_header


def _feature_shape(models):
    try:
        shape = models["feature_extractor"].output_shape
        return "×".join(str(d) for d in shape if d is not None)
    except Exception:
        return "n/a"


def render(ctx):
    models = ctx["models"]
    h, w = models["image_size"]
    classes = ", ".join(f"{name} ({idx})" for name, idx in models["class_indices"].items())

    page_header("About", "Objective, model design and inference workflow.")

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        st.markdown("## Objective")
        st.markdown(
            "Identify driver drowsiness from facial images. Given a frame from a camera, an image "
            "or a video, the system locates the driver's face and predicts whether the driver "
            "appears drowsy."
        )

        st.markdown("## MobileNetV2 feature extraction")
        st.markdown(
            "MobileNetV2 is a lightweight convolutional neural network designed for efficient "
            "inference. Here it is used with its weights frozen as a feature extractor: it is "
            "not trained or fine-tuned inside this application. It converts a face image into a "
            "numeric feature vector that summarises its visual content."
        )

        st.markdown("## XGBoost classification")
        st.markdown(
            "XGBoost is a gradient-boosted decision-tree classifier. It receives the feature "
            "vector and outputs a probability for each class. The predicted class is the one with "
            "the highest probability, and that probability is shown as the confidence score."
        )

        st.markdown("## Inference workflow")
        st.markdown(
            "1. **Frame** is captured from the webcam, image or video.\n"
            "2. **Face detection** with an OpenCV Haar cascade selects the largest face.\n"
            f"3. **Preprocessing** converts the crop BGR → RGB, resizes it to {w}×{h} "
            "(area interpolation) and scales pixel values to 0–1.\n"
            "4. **MobileNetV2** produces the feature vector.\n"
            "5. **XGBoost** returns class probabilities, from which the prediction is derived."
        )

        st.markdown("## Limitations")
        st.markdown(
            "Predictions are made on single frames, without temporal smoothing. Accuracy depends "
            "on lighting, camera angle, occlusion (for example glasses or masks) and how closely "
            "the input resembles the training data. This application is not a certified safety system."
        )

    with right:
        st.markdown("## Loaded model")
        st.markdown(
            '<div class="dds-card">'
            + kv_rows(
                [
                    ("Feature extractor", "MobileNetV2"),
                    ("Classifier", "XGBoost"),
                    ("Input size", f"{w} × {h} × 3 (RGB)"),
                    ("Feature output shape", _feature_shape(models)),
                    ("Classes (index)", classes),
                    ("Face detector", "OpenCV Haar cascade"),
                ]
            )
            + "</div>",
            unsafe_allow_html=True,
        )
