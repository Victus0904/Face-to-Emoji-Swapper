# 🤖 Face to Emoji Swapper

[![CI](https://github.com/Victus0904/Face-to-Emoji-Swapper/actions/workflows/ci.yml/badge.svg)](https://github.com/Victus0904/Face-to-Emoji-Swapper/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Streamlit app that finds every face in a photo with OpenCV and covers it with the
emoji of your choice. The emoji doesn't need a transparent background: a soft circular
mask blends it into the photo.

| Input | Output |
|---|---|
| ![input](https://github.com/user-attachments/assets/c6ae4567-73b9-4278-a93e-aa4d25cd1fa6) | ![output](https://github.com/user-attachments/assets/b4e4b1bf-b849-44a7-8317-73413706982d) |

## Features

- **Multi-face detection** with OpenCV's Haar cascade frontal-face model
- **Alpha-free blending.** A feathered circular mask, so any JPG or PNG emoji works.
- **Edge-safe.** Faces touching the image border are clipped correctly.
- **Adjustable** emoji size and detection strictness
- **One-click download** of the result as PNG

## Quick start

```bash
git clone https://github.com/Victus0904/Face-to-Emoji-Swapper.git
cd Face-to-Emoji-Swapper
pip install -r requirements.txt
streamlit run app.py
```

## Use it from Python

```python
import cv2
from swapper import swap_faces

photo = cv2.imread("group.jpg")
emoji = cv2.imread("smile.png")
result, n_faces = swap_faces(photo, emoji, scale=1.15)
cv2.imwrite("result.png", result)
```

## How it works

1. The photo is converted to grayscale and passed to `CascadeClassifier.detectMultiScale`.
2. For each `(x, y, w, h)` box the emoji is resized (optionally scaled around the box centre).
3. A Gaussian-feathered circular mask `m` blends the pixels: `out = roi·(1 − m) + emoji·m`.

## Project structure

```text
.
├── app.py              # Streamlit UI
├── swapper.py          # Detection + blending (no UI dependencies)
├── tests/              # Unit tests and a headless app smoke test
└── requirements.txt
```

## Running the tests

```bash
pip install pytest
pytest -q
```

## License

[MIT](LICENSE)
