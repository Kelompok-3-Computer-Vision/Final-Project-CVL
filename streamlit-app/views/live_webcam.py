"""Live Webcam page (Streamlit WebRTC)."""

import time

import streamlit as st

from settings import DEFAULT_MAX_INFERENCE_FPS, STALE_RESULT_SECONDS
from theme import chip, fmt_ms, kv_rows, metric_card, page_header, placeholder, prediction_card

RESOLUTIONS = {
    "640 × 480 (recommended)": (640, 480),
    "1280 × 720": (1280, 720),
}
DEFAULT_RESOLUTION = "640 × 480 (recommended)"

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def _constraints(width, height):
    # Ask the browser for a modest resolution and a capped frame rate: fewer pixels to
    # encode/decode and to scan for faces, and no point capturing faster than we display.
    return {
        "video": {
            "width": {"ideal": width},
            "height": {"ideal": height},
            "frameRate": {"ideal": 30, "max": 30},
        },
        "audio": False,
    }


def _prediction_view(snap, playing):
    """-> (label, tone, chip text, prob_drowsy, confidence)"""
    if not playing:
        return "—", "neutral", "Camera off", None, None
    if snap["error"]:
        return "Inference error", "alert", "Error", None, None
    if not snap["has_result"]:
        return "Starting…", "accent", "Waiting for first frame", None, None
    if snap["age_s"] is not None and snap["age_s"] > STALE_RESULT_SECONDS:
        return "No recent result", "neutral", "Stale", None, None
    if not snap["face"]:
        return "No face detected", "neutral", "No face", None, None
    r = snap["result"]
    if r["is_drowsy"]:
        return r["label"], "alert", "Drowsiness detected", r["prob_drowsy"], r["confidence"]
    return r["label"], "ok", "Normal", r["prob_drowsy"], r["confidence"]


def _render_metrics(snap, playing, metrics_ph, breakdown_ph):
    label, tone, chip_text, p_drowsy, conf = _prediction_view(snap, playing)
    fps = f"{snap['fps']:.1f}" if playing and snap["fps"] else "—"
    lat = fmt_ms(snap["latency_ms"]) if playing else "—"
    rate = f"{snap['inference_rate']:.1f}" if playing and snap["inference_rate"] else "—"

    html = prediction_card(label, tone, chip_text, p_drowsy, conf)
    html += (
        '<div class="dds-grid2">'
        + metric_card("FPS", fps, "fps", f"Analysis {rate}/s")
        + metric_card("Latency", lat, "ms", "Inference, last 30 avg")
        + "</div>"
    )
    if snap["error"]:
        html += f'<div class="dds-note" style="margin-top:.6rem;color:#C8403D">{snap["error"]}</div>'
    metrics_ph.markdown(html, unsafe_allow_html=True)

    b = snap["breakdown_ms"]
    rows = [(k, f"{v:.1f} ms" if v is not None else "—") for k, v in b.items()]
    rows.append(("Pipeline total", f"{snap['latency_ms']:.1f} ms" if snap["latency_ms"] is not None else "—"))
    rows.append(("Frame-to-result lag", f"{snap['lag_ms']:.0f} ms" if snap["lag_ms"] is not None else "—"))
    breakdown_ph.markdown(kv_rows(rows), unsafe_allow_html=True)


def render(ctx):
    page_header(
        "Live Webcam",
        "Real-time drowsiness detection from your camera. Frames are analysed by the server "
        "running this app; nothing is stored.",
    )

    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        from webcam_processor import DrowsinessProcessor  # imports av / streamlit_webrtc
    except ImportError:
        st.error(
            "The `streamlit-webrtc` and `av` packages are not installed.\n\n"
            "Install them with: `pip install streamlit-webrtc av`"
        )
        return

    models, cascade = ctx["models"], ctx["cascade"]

    # Widget values from the Advanced panel (rendered below) are already in session_state
    # when the script reruns, so they can configure the stream created above them.
    res_key = st.session_state.get("cam_resolution", DEFAULT_RESOLUTION)
    width, height = RESOLUTIONS.get(res_key, RESOLUTIONS[DEFAULT_RESOLUTION])
    max_fps = float(st.session_state.get("cam_max_fps", DEFAULT_MAX_INFERENCE_FPS))

    left, right = st.columns([2.1, 1], gap="large")

    with left:
        status_ph = st.empty()
        preview_ph = st.empty()
        stream_kwargs = dict(
            key="drowsiness-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=_constraints(width, height),
            video_processor_factory=lambda: DrowsinessProcessor(models, cascade, max_fps),
            async_processing=True,
            video_html_attrs={
                "style": {"width": "100%", "borderRadius": "12px", "background": "#0F172A"},
                "controls": False,
                "autoPlay": True,
                "muted": True,
                "playsInline": True,
            },
        )
        try:
            webrtc_ctx = webrtc_streamer(
                **stream_kwargs, translations={"start": "Start camera", "stop": "Stop camera"}
            )
        except TypeError:  # older streamlit-webrtc without the `translations` option
            webrtc_ctx = webrtc_streamer(**stream_kwargs)
        with st.expander("Advanced settings"):
            st.selectbox(
                "Camera resolution", list(RESOLUTIONS), key="cam_resolution",
                help="Applied the next time the camera is started.",
            )
            st.slider(
                "Maximum analysis rate (per second)", 1, 15, int(DEFAULT_MAX_INFERENCE_FPS), key="cam_max_fps",
                help="Upper limit on how often a frame is analysed. The video preview always runs "
                     "at the camera's frame rate; lower values leave more CPU for it.",
            )
        st.caption(
            "If the video does not appear, open the app via `localhost` or HTTPS — browsers block "
            "camera access on plain HTTP connections to other hosts."
        )

    with right:
        metrics_ph = st.empty()
        with st.expander("Timing breakdown"):
            breakdown_ph = st.empty()

    state = webrtc_ctx.state
    playing = bool(state.playing)
    proc = webrtc_ctx.video_processor if playing else None

    # Static states
    if playing:
        preview_ph.empty()
        status_ph.markdown(
            '<div style="margin-bottom:.6rem">' + chip("Streaming", "ok") + "</div>", unsafe_allow_html=True
        )
    else:
        if state.signalling:
            status_ph.markdown(
                '<div style="margin-bottom:.6rem">' + chip("Connecting", "accent") + "</div>",
                unsafe_allow_html=True,
            )
            preview_ph.markdown(
                placeholder("Connecting to camera", "Allow camera access in your browser if prompted."),
                unsafe_allow_html=True,
            )
        else:
            status_ph.markdown(
                '<div style="margin-bottom:.6rem">' + chip("Camera off") + "</div>", unsafe_allow_html=True
            )
            preview_ph.markdown(
                placeholder(
                    "Camera is off",
                    "Select “Start camera” below and allow access when your browser asks.",
                ),
                unsafe_allow_html=True,
            )

    empty_snap = {
        "fps": 0.0, "inference_rate": 0.0, "latency_ms": None, "lag_ms": None, "age_s": None,
        "has_result": False, "face": False, "result": None, "error": None, "frame_size": None,
        "breakdown_ms": {"Face detection": None, "Preprocessing": None, "MobileNetV2": None, "XGBoost": None},
    }

    if proc is None:
        _render_metrics(empty_snap, False, metrics_ph, breakdown_ph)
        return

    # Live metrics loop. The script is interrupted (and rerun) by Streamlit as soon as the
    # user presses Stop or navigates elsewhere, so this loop needs no explicit exit.
    proc.set_max_inference_fps(max_fps)
    while not proc.ended:
        snap = proc.get_snapshot()
        size = snap["frame_size"]
        status_ph.markdown(
            '<div style="margin-bottom:.6rem">'
            + chip("Streaming" + (f" · {size[0]}×{size[1]}" if size else ""), "ok")
            + "</div>",
            unsafe_allow_html=True,
        )
        _render_metrics(snap, True, metrics_ph, breakdown_ph)
        time.sleep(0.25)
