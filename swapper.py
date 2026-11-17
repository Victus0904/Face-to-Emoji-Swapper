"""Face detection and emoji blending, independent of the Streamlit UI."""

from __future__ import annotations

import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's pretrained frontal-face Haar cascade."""
    detector = cv2.CascadeClassifier(_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {_CASCADE_PATH}")
    return detector


def detect_faces(
    image_bgr: np.ndarray,
    detector: cv2.CascadeClassifier,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> list[tuple[int, int, int, int]]:
    """Return ``(x, y, w, h)`` boxes for every face found in a BGR image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return [tuple(int(v) for v in box) for box in faces]


def overlay_emoji(
    frame: np.ndarray, emoji: np.ndarray, x: int, y: int, w: int, h: int, scale: float = 1.0
) -> np.ndarray:
    """Blend ``emoji`` over the box ``(x, y, w, h)`` of ``frame`` through a circular mask.

    The emoji does not need an alpha channel: a soft-edged circle decides which pixels
    are replaced. ``scale`` grows or shrinks the emoji around the box centre. The box is
    clipped to the frame, so faces near the edge are handled. Returns a new array.
    """
    out = frame.copy()
    cx, cy = x + w / 2, y + h / 2
    w, h = max(1, int(w * scale)), max(1, int(h * scale))
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, out.shape[1]), min(y + h, out.shape[0])
    if x0 >= x1 or y0 >= y1:
        return out

    emoji_resized = cv2.resize(emoji[:, :, :3], (w, h), interpolation=cv2.INTER_AREA)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, min(w, h) / 60))[:, :, None]

    # Crop the emoji/mask to the visible part of the box.
    ex0, ey0 = x0 - x, y0 - y
    ex1, ey1 = ex0 + (x1 - x0), ey0 + (y1 - y0)
    m = mask[ey0:ey1, ex0:ex1]
    roi = out[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1 - m) + emoji_resized[ey0:ey1, ex0:ex1].astype(np.float32) * m
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def swap_faces(
    image_bgr: np.ndarray,
    emoji_bgr: np.ndarray,
    detector: cv2.CascadeClassifier | None = None,
    scale: float = 1.0,
    min_neighbors: int = 5,
) -> tuple[np.ndarray, int]:
    """Replace every detected face with the emoji. Returns ``(result, n_faces)``."""
    detector = detector or load_face_detector()
    faces = detect_faces(image_bgr, detector, min_neighbors=min_neighbors)
    result = image_bgr
    for x, y, w, h in faces:
        result = overlay_emoji(result, emoji_bgr, x, y, w, h, scale=scale)
    return (result if faces else image_bgr.copy()), len(faces)
