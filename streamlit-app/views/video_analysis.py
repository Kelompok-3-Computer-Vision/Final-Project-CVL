"""Video Analysis page."""

import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

from model_utils import detect_largest_face, predict_drowsiness_timed
from overlay import annotate_frame
from theme import chip, kv_rows, metric_card, page_header, show_image


def _summary_html(s):
    drowsy_pct = (s["drowsy_frames"] / s["with_face"] * 100) if s["with_face"] else 0.0
    verdict_tone = "alert" if s["drowsy_frames"] else "ok"
    verdict = "Drowsiness detected in this video" if s["drowsy_frames"] else "No drowsiness detected"
    if s["with_face"] == 0:
        verdict_tone, verdict = "neutral", "No face detected in analyzed frames"
    grid = (
        '<div class="dds-grid2" style="grid-template-columns:repeat(4,1fr)">'
        + metric_card("Frames analyzed", f"{s['analyzed']}", "", f"of {s['total_frames']} total" if s["total_frames"] else "")
        + metric_card("Drowsy frames", f"{drowsy_pct:.1f}", "%", f"{s['drowsy_frames']} of {s['with_face']} with a face")
        + metric_card("Peak drowsy probability", f"{s['max_prob'] * 100:.1f}", "%")
        + metric_card("Mean inference latency", f"{s['mean_latency_ms']:.0f}", "ms", f"{s['proc_fps']:.1f} frames analyzed / s")
        + "</div>"
    )
    return (
        '<div class="dds-card" style="margin-bottom:.9rem"><div style="display:flex;justify-content:space-between;'
        'align-items:center"><div class="dds-label">Result</div>' + chip(verdict, verdict_tone) + "</div></div>" + grid
    )


def _render_summary(s):
    st.markdown(_summary_html(s), unsafe_allow_html=True)
    if s["prob_series"]:
        st.markdown(
            '<div class="dds-label" style="margin:1.1rem 0 .3rem">Drowsy probability by frame</div>',
            unsafe_allow_html=True,
        )
        st.line_chart({"Drowsy probability": s["prob_series"]}, color="#2F5FD0", height=220)


def render(ctx):
    models, cascade = ctx["models"], ctx["cascade"]
    page_header(
        "Video Analysis",
        "Upload a driving video. Every Nth frame is analyzed and the result is summarized below.",
    )

    uploaded = st.file_uploader(
        "Drag and drop a video here, or browse", type=["mp4", "avi", "mov", "mkv"], key="video_upload"
    )
    if uploaded is None:
        st.session_state.pop("video_summary", None)
        return

    left, right = st.columns([1.7, 1], gap="large")
    with right:
        frame_skip = st.slider(
            "Analyze every N frames", 1, 10, 3,
            help="Higher values process faster but sample the video less often.",
        )
        run = st.button("Start analysis", type="primary", key="video_run")
        st.markdown('<div class="dds-label" style="margin:1.2rem 0 .3rem">Progress</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_ph = st.empty()
        metrics_ph = st.empty()
    with left:
        preview_ph = st.empty()
        with preview_ph.container():
            st.video(uploaded.getvalue())

    file_id = f"{uploaded.name}:{uploaded.size}:{frame_skip}"

    if not run:
        summary = st.session_state.get("video_summary")
        if summary and summary["file_id"] == file_id:
            progress_bar.progress(1.0)
            st.markdown("---")
            _render_summary(summary)
        return

    st.session_state.pop("video_summary", None)

    # OpenCV needs a file path, so the upload is written to a temporary file (removed afterwards).
    suffix = os.path.splitext(uploaded.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    try:
        if not cap.isOpened():
            st.error("This video could not be opened. Try another file or format.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = analyzed = with_face = drowsy_frames = 0
        max_prob = 0.0
        latencies, prob_series = [], []
        t_start = time.perf_counter()
        last_ui = 0.0

        while True:
            if not cap.grab():          # cheap skip for frames that are not analyzed
                break
            frame_idx += 1
            if frame_idx % frame_skip:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break

            t0 = time.perf_counter()
            face_box = detect_largest_face(frame, cascade)
            result = None
            if face_box is not None:
                x, y, w, h = face_box
                result = predict_drowsiness_timed(frame[y:y + h, x:x + w], models)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            analyzed += 1

            if result is not None:
                with_face += 1
                drowsy_frames += int(result["is_drowsy"])
                max_prob = max(max_prob, result["prob_drowsy"])
                prob_series.append(round(result["prob_drowsy"], 4))
                annotate_frame(frame, face_box, result)
            else:
                prob_series.append(None)

            # Update the UI at most ~10x per second so rendering never dominates processing time.
            now = time.perf_counter()
            if now - last_ui >= 0.1:
                last_ui = now
                with preview_ph.container():
                    show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                if result is None:
                    status_ph.markdown(chip("No face in this frame"), unsafe_allow_html=True)
                elif result["is_drowsy"]:
                    status_ph.markdown(
                        chip(f"Drowsy · {result['confidence'] * 100:.1f}%", "alert"), unsafe_allow_html=True
                    )
                else:
                    status_ph.markdown(
                        chip(f"Not drowsy · {result['confidence'] * 100:.1f}%", "ok"), unsafe_allow_html=True
                    )
                elapsed = now - t_start
                metrics_ph.markdown(
                    '<div class="dds-card" style="margin-top:.9rem">'
                    + kv_rows(
                        [
                            ("Frames analyzed", str(analyzed)),
                            ("Analysis speed", f"{analyzed / elapsed:.1f} frames/s"),
                            ("Mean inference latency", f"{np.mean(latencies):.0f} ms"),
                            ("Peak drowsy probability", f"{max_prob * 100:.1f}%"),
                        ]
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
    finally:
        cap.release()
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    elapsed = max(time.perf_counter() - t_start, 1e-6)
    with preview_ph.container():   # put the original video player back
        st.video(uploaded.getvalue())
    progress_bar.progress(1.0)
    status_ph.empty()
    metrics_ph.empty()
    summary = {
        "file_id": file_id,
        "analyzed": analyzed,
        "with_face": with_face,
        "drowsy_frames": drowsy_frames,
        "max_prob": max_prob,
        "total_frames": total_frames if total_frames > 0 else frame_idx,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "proc_fps": analyzed / elapsed,
        "prob_series": prob_series,
    }
    st.session_state["video_summary"] = summary
    st.markdown("---")
    _render_summary(summary)
