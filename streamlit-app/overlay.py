"""Frame annotation (face box + label). Pure OpenCV, no Streamlit dependency."""

import cv2

# BGR
COLOR_DROWSY = (69, 69, 214)      # muted red
COLOR_ALERT = (138, 157, 15)      # teal


def annotate_frame(frame_bgr, face_box, result):
    """Draw the face box and a label chip on frame_bgr (in place). Returns (frame, is_drowsy)."""
    x, y, w, h = face_box
    is_drowsy = result["is_drowsy"]
    color = COLOR_DROWSY if is_drowsy else COLOR_ALERT
    label = f"{result['label']}  {result['confidence'] * 100:.0f}%"

    # Scale text/line sizes with the image so large photos stay legible.
    fh, fw = frame_bgr.shape[:2]
    scale = max(0.5, min(fw, 2000) / 900.0)
    thick = max(1, int(round(scale * 1.6)))
    box_thick = max(2, int(round(scale * 2.2)))

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, box_thick)

    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(6 * scale)
    top = max(y - th - base - 2 * pad, 0)
    cv2.rectangle(frame_bgr, (x, top), (x + tw + 2 * pad, top + th + base + 2 * pad), color, -1)
    cv2.putText(
        frame_bgr, label, (x + pad, top + th + pad),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA,
    )
    return frame_bgr, is_drowsy
