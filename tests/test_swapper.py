import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swapper import load_face_detector, overlay_emoji, swap_faces  # noqa: E402


def test_overlay_replaces_centre_and_keeps_corners():
    frame = np.zeros((100, 100, 3), np.uint8)
    emoji = np.full((50, 50, 3), 255, np.uint8)
    out = overlay_emoji(frame, emoji, 25, 25, 50, 50)
    assert out.shape == frame.shape
    assert out[50, 50].min() > 200          # centre of the circle is the emoji
    assert out[26, 26].max() < 50           # box corner lies outside the circle
    assert frame.max() == 0                 # input is not modified in place


def test_overlay_clips_boxes_at_the_border():
    frame = np.zeros((60, 60, 3), np.uint8)
    emoji = np.full((10, 10, 3), 255, np.uint8)
    out = overlay_emoji(frame, emoji, 40, 40, 40, 40)
    assert out.shape == frame.shape
    assert out[59, 59].min() > 0
    assert overlay_emoji(frame, emoji, 200, 200, 10, 10).max() == 0


def test_swap_faces_without_faces_returns_copy():
    blank = np.zeros((120, 120, 3), np.uint8)
    emoji = np.full((20, 20, 3), 255, np.uint8)
    result, n = swap_faces(blank, emoji, load_face_detector())
    assert n == 0
    assert np.array_equal(result, blank)


def test_app_smoke():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not at.exception
    assert len(at.info) == 1
