"""
WebRTC video processor for the live webcam page.

Design (why the preview stays smooth):
  * recv() runs on the WebRTC path and only does cheap work: it hands the newest
    frame to a worker, draws the *latest available* result onto a copy and
    returns. It never runs face detection or model inference.
  * A single worker thread does detection + MobileNetV2 + XGBoost. It always
    picks up the most recent frame (older ones are overwritten, never queued),
    never processes the same frame twice, and is rate-limited so it cannot starve
    the video path of CPU.
  * All FPS / latency numbers are measured from real timestamps.

Stabilisation (why predictions no longer flip rapidly):
  * Temporal smoothing: prob_drowsy is averaged over the last SMOOTHING_WINDOW
    inferences (rolling mean) before the label is decided.
  * Hysteresis thresholds: switching from Non-Drowsy → Drowsy requires the
    smoothed probability to reach HYSTERESIS_HIGH; the reverse requires it to
    drop to HYSTERESIS_LOW. Values between the two thresholds keep the previous
    label, eliminating rapid toggling near the decision boundary.
  * When no face is detected the probability buffer is cleared so the next
    detection starts fresh rather than inheriting stale history.

No Streamlit UI calls are made here; the UI reads get_snapshot().
"""

import threading
import time
from collections import deque

import av
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from model_utils import detect_largest_face, predict_drowsiness_timed
from overlay import annotate_frame
from settings import (
    DEFAULT_MAX_INFERENCE_FPS,
    FPS_WINDOW,
    HYSTERESIS_HIGH,
    HYSTERESIS_LOW,
    SMOOTHING_WINDOW,
    STALE_RESULT_SECONDS,
)


def _rate(timestamps, now, max_age=2.0):
    """Events per second over the recorded window; 0 when too few/too old."""
    if len(timestamps) < 2 or now - timestamps[-1] > max_age:
        return 0.0
    span = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / span if span > 0 else 0.0


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self, models, cascade, max_inference_fps=DEFAULT_MAX_INFERENCE_FPS):
        self._models = models
        self._cascade = cascade
        self.max_inference_fps = max_inference_fps

        # newest frame slot (single slot: overwritten, never queued)
        self._cond = threading.Condition()
        self._latest_img = None
        self._latest_id = 0
        self._latest_ts = 0.0

        # published result + stats
        self._state_lock = threading.Lock()
        self._published = None          # dict, see _publish()
        self._out_ts = deque(maxlen=FPS_WINDOW)   # output frame timestamps
        self._inf_ts = deque(maxlen=FPS_WINDOW)   # inference completion timestamps
        self._timings = deque(maxlen=FPS_WINDOW)  # per-inference timing dicts
        self._frames_in = 0
        self._frame_size = None         # (width, height) of the incoming camera frames
        self._error = None

        # --- Temporal smoothing & hysteresis state ---
        # Rolling buffer of raw prob_drowsy values (one per inference with a detected face)
        self._prob_buffer = deque(maxlen=SMOOTHING_WINDOW)
        # Current hysteresis state; starts as Non-Drowsy until evidence accumulates
        non_drowsy_candidates = [
            k for k in models["class_indices"] if "non" in k.lower()
        ]
        self._hysteresis_state = non_drowsy_candidates[0] if non_drowsy_candidates else list(models["class_indices"])[0]

        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="drowsiness-worker", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ video path
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        now = time.perf_counter()
        img = frame.to_ndarray(format="bgr24")

        # Hand the frame to the worker. The worker only reads this array and we
        # never draw on it, so no copy is needed here.
        with self._cond:
            self._latest_img = img
            self._latest_id += 1
            self._latest_ts = now
            self._frames_in += 1
            self._frame_size = (img.shape[1], img.shape[0])
            self._cond.notify()

        with self._state_lock:
            self._out_ts.append(now)
            pub = self._published

        if pub is not None and pub["box"] is not None and now - pub["done_ts"] < STALE_RESULT_SECONDS:
            out = img.copy()  # draw on a copy: `img` is shared with the worker
            annotate_frame(out, pub["box"], pub["result"])
            return av.VideoFrame.from_ndarray(out, format="bgr24")

        return frame  # nothing to draw: pass the original frame through untouched

    def on_ended(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------------ worker
    def _run(self):
        last_id = 0
        while not self._stop.is_set():
            with self._cond:
                while self._latest_id == last_id and not self._stop.is_set():
                    self._cond.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                img, last_id, arrived = self._latest_img, self._latest_id, self._latest_ts

            started = time.perf_counter()
            try:
                self._process(img, arrived, started)
                self._error = None
            except Exception as exc:  # keep the worker alive; surface the error in the UI
                self._error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(0.25)
                continue

            # Rate limit: leave CPU for the video path.
            min_interval = 1.0 / max(self.max_inference_fps, 0.5)
            remaining = min_interval - (time.perf_counter() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _apply_smoothing_and_hysteresis(self, raw_result):
        """
        Apply temporal smoothing (rolling mean of prob_drowsy over the last
        SMOOTHING_WINDOW inferences) and hysteresis thresholding to prevent
        rapid label flipping near the decision boundary.

        Returns a *new* result dict with stabilised label, is_drowsy,
        prob_drowsy, and confidence fields; all other keys are unchanged.
        """
        self._prob_buffer.append(raw_result["prob_drowsy"])
        smoothed_prob = float(np.mean(self._prob_buffer))

        drowsy_label = self._models["drowsy_label"]
        # Determine the Non-Drowsy label dynamically (never hardcode the string)
        non_drowsy_label = next(
            (k for k in self._models["class_indices"] if "non" in k.lower()),
            drowsy_label,  # fallback: same label (edge case)
        )

        # Hysteresis: only switch state when the smoothed probability clearly
        # crosses a threshold, not just nudges past 0.5.
        if smoothed_prob >= HYSTERESIS_HIGH:
            self._hysteresis_state = drowsy_label
        elif smoothed_prob <= HYSTERESIS_LOW:
            self._hysteresis_state = non_drowsy_label
        # else: keep self._hysteresis_state unchanged (the "dead zone")

        stabilised_is_drowsy = self._hysteresis_state == drowsy_label
        stabilised_label = self._hysteresis_state
        stabilised_confidence = smoothed_prob if stabilised_is_drowsy else (1.0 - smoothed_prob)

        return {
            **raw_result,
            "label": stabilised_label,
            "is_drowsy": stabilised_is_drowsy,
            "prob_drowsy": smoothed_prob,       # smoothed value shown in UI
            "confidence": stabilised_confidence,
            "raw_prob_drowsy": raw_result["prob_drowsy"],  # original value for debugging
        }

    def _process(self, img, arrived, started):
        t0 = time.perf_counter()
        box = detect_largest_face(img, self._cascade)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        result = None
        if box is not None:
            x, y, w, h = box
            raw_result = predict_drowsiness_timed(img[y:y + h, x:x + w], self._models)
            result = self._apply_smoothing_and_hysteresis(raw_result)
        else:
            # No face detected: clear the probability buffer so stale history
            # does not influence the next detection.
            self._prob_buffer.clear()

        done = time.perf_counter()
        timing = {"detect": detect_ms}
        if result is not None:
            timing.update(result["timings_ms"])
            timing["pipeline"] = detect_ms + result["timings_ms"]["total"]
        else:
            timing["pipeline"] = detect_ms

        with self._state_lock:
            self._published = {
                "box": box,
                "result": result,
                "done_ts": done,
                "lag_ms": (done - arrived) * 1000.0,   # camera frame arrival -> result ready
            }
            self._inf_ts.append(done)
            self._timings.append(timing)

    # ------------------------------------------------------------------ UI-facing
    def set_max_inference_fps(self, value):
        self.max_inference_fps = float(value)

    @property
    def ended(self):
        return self._stop.is_set()

    def get_snapshot(self):
        """Thread-safe copy of the current measurements for the UI."""
        now = time.perf_counter()
        with self._state_lock:
            pub = self._published
            timings = list(self._timings)
            fps = _rate(self._out_ts, now)
            inf_rate = _rate(self._inf_ts, now)
        with self._cond:
            frames_in = self._frames_in
            frame_size = self._frame_size

        def mean(key):
            vals = [t[key] for t in timings if key in t]
            return float(np.mean(vals)) if vals else None

        snap = {
            "frames_in": frames_in,
            "frame_size": frame_size,
            "fps": fps,
            "inference_rate": inf_rate,
            "latency_ms": mean("pipeline"),
            "breakdown_ms": {
                "Face detection": mean("detect"),
                "Preprocessing": mean("preprocess"),
                "MobileNetV2": mean("features"),
                "XGBoost": mean("xgboost"),
            },
            "lag_ms": pub["lag_ms"] if pub else None,
            "age_s": (now - pub["done_ts"]) if pub else None,
            "has_result": pub is not None,
            "face": False,
            "result": None,
            "error": self._error,
        }
        if pub is not None and pub["box"] is not None and pub["result"] is not None:
            snap["face"] = True
            snap["result"] = pub["result"]
        return snap
