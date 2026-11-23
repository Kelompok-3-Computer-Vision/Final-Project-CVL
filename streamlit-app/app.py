"""
Drowsiness Detection System - Streamlit app entry point.
Model: MobileNetV2 (frozen feature extractor) + XGBoost.

Run with:  streamlit run app.py

Layout:
  app.py               entry point: page config, sidebar navigation, model loading, routing
  model_utils.py       model loading (cached) + inference (no UI code)
  webcam_processor.py  WebRTC processor: lightweight callback + inference worker thread
  overlay.py           face box / label drawing
  theme.py             light-theme CSS and HTML helpers
  nav.py               page registry
  views/               one module per page
"""

import streamlit as st

st.set_page_config(
    page_title="Drowsiness Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the WebRTC stack (aiortc / libsrtp / OpenSSL) BEFORE TensorFlow. On some platform and
# package-version combinations the process segfaults if TensorFlow's native libraries are
# loaded first. The import is optional: the Live Webcam page reports it if missing.
try:
    import streamlit_webrtc  # noqa: F401, E402
except ImportError:
    pass

from model_utils import MODEL_DIR, load_face_cascade, load_models  # noqa: E402
from nav import PAGES, current_page, go  # noqa: E402
from theme import inject_css  # noqa: E402
from views import about, image_analysis, live_webcam, overview, video_analysis  # noqa: E402

inject_css()

ROUTES = {
    "Overview": overview.render,
    "Live Webcam": live_webcam.render,
    "Image Analysis": image_analysis.render,
    "Video Analysis": video_analysis.render,
    "About": about.render,
}

LOGO_SVG = (
    '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="30" height="30" rx="7" fill="#2F5FD0"/>'
    '<path d="M5.5 15c2.2-3.6 5.3-5.4 9.5-5.4s7.3 1.8 9.5 5.4c-2.2 3.6-5.3 5.4-9.5 5.4S7.7 18.6 5.5 15z" '
    'stroke="#fff" stroke-width="1.8" stroke-linejoin="round"/>'
    '<circle cx="15" cy="15" r="2.6" fill="#fff"/></svg>'
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="dds-brand">{LOGO_SVG}<div><div class="name">Drowsiness Detection System</div>'
        '<div class="sub">Driver monitoring</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="dds-side-label">Navigation</div>', unsafe_allow_html=True)
    active = current_page()
    for page in PAGES:
        st.button(
            page,
            key=f"nav_{page}",
            type="primary" if page == active else "secondary",
            use_container_width=True,
            on_click=go,
            args=(page,),
        )

    st.markdown(
        '<div class="dds-side-foot"><b>Model:</b> MobileNetV2 + XGBoost</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Model folder"):
        model_dir = st.text_input(
            "Folder path",
            value=MODEL_DIR,
            label_visibility="collapsed",
            help="Folder containing mobilenetv2_feature_extractor.keras, "
                 "xgb_drowsiness_model.json and metadata.json.",
        )

# ---------------------------------------------------------------------------
# Models (cached by st.cache_resource: loaded once per process, not per rerun/frame)
# ---------------------------------------------------------------------------
try:
    models = load_models(model_dir)
    face_cascade = load_face_cascade()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Run the 'Menyimpan Model (Usulan: MobileNetV2 + XGBoost)' cell in the Kaggle notebook, "
        "download the resulting `.zip`, extract it and place the folder next to `app.py` "
        "(or set the path under 'Model folder' in the sidebar)."
    )
    st.stop()

ROUTES[active]({"models": models, "cascade": face_cascade})
