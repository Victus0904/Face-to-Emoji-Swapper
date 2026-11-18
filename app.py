"""Streamlit front end for the face-to-emoji swapper.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from swapper import load_face_detector, swap_faces


@st.cache_resource(show_spinner=False)
def get_detector():
    return load_face_detector()


def to_bgr(uploaded) -> np.ndarray:
    """Decode an uploaded image file into an OpenCV BGR array."""
    rgb = np.array(Image.open(uploaded).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    st.set_page_config(page_title="Face to Emoji Swapper", page_icon="🤖", layout="wide")
    st.title("🤖 Face to Emoji Swapper")
    st.caption("Detects every face in a photo and covers it with the emoji of your choice.")

    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Upload a photo with one or more faces.\n"
            "2. Upload an emoji image (no transparency needed).\n"
            "3. Click **Swap faces**."
        )
        st.header("Settings")
        scale = st.slider("Emoji size", 0.8, 1.6, 1.15, 0.05)
        min_neighbors = st.slider(
            "Detection strictness", 3, 10, 5, help="Higher means fewer false detections."
        )

    col1, col2 = st.columns(2)
    face_file = col1.file_uploader("Photo", type=["jpg", "jpeg", "png"])
    emoji_file = col2.file_uploader("Emoji", type=["jpg", "jpeg", "png"])

    if not (face_file and emoji_file):
        st.info("Upload both a photo and an emoji image to get started.")
        return

    photo = to_bgr(face_file)
    emoji = to_bgr(emoji_file)
    col1.image(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
    col2.image(cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB), caption="Emoji", width=160)

    if st.button("Swap faces", type="primary"):
        result, n_faces = swap_faces(
            photo, emoji, get_detector(), scale=scale, min_neighbors=min_neighbors
        )
        if n_faces == 0:
            st.warning("No faces detected. Try a clearer, front-facing photo or lower the strictness.")
            return
        st.success(f"Replaced {n_faces} face(s).")
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Result", width="stretch")

        buf = io.BytesIO()
        Image.fromarray(result_rgb).save(buf, format="PNG")
        st.download_button("Download result", buf.getvalue(), "emoji_swap.png", "image/png")


if __name__ == "__main__":
    main()
