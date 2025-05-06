#!/usr/bin/env python
# coding: utf-8

# In[33]:


import cv2 
import numpy as np
import streamlit as st
from PIL import Image


# In[ ]:


#jupyter nbconvert Untitled-1.ipynb --to python


# In[29]:


# Set app title
st.title("🤖 Face to Emoji Swapper")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Upload a face image (JPG/PNG).\n\n"
    "2. Upload an emoji image (JPG/PNG, no alpha needed).\n\n"
    "3. Click 'Swap Faces' to see the result!"
)


# In[28]:


# Upload face image
face_file = st.file_uploader("Upload a Face Image", type=["jpg", "jpeg", "png"])
emoji_file = st.file_uploader("Upload an Emoji Image", type=["jpg", "jpeg", "png"])


# In[ ]:





# In[13]:


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[31]:


def overlay_emoji_no_alpha(frame, emoji_file, x, y, w, h):
    # Resize emoji to match face size
    emoji_resized = cv2.resize(emoji_file, (w, h))

    # Create a circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(w, h)//2, 255, -1)
    mask = mask.astype(float) / 255.0  # Normalize to [0, 1]

    # Repeat mask for RGB channels
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Get ROI from frame
    roi = frame[y:y+h, x:x+w].astype(float)

    # Perform blending
    blended = roi * (1 - mask_rgb) + emoji_resized.astype(float) * mask_rgb

    # Replace in frame
    frame[y:y+h, x:x+w] = blended.astype(np.uint8)
    return frame


# In[32]:


if face_file and emoji_file:
    # Read uploaded images
    face_img = Image.open(face_file).convert('RGB')
    emoji_img = Image.open(emoji_file).convert('RGB')

    # Convert to OpenCV BGR
    face_np = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    emoji_np = cv2.cvtColor(np.array(emoji_img), cv2.COLOR_RGB2BGR)

    st.image(face_img, caption="Original Face Image", use_column_width=True)

    if st.button("Swap Faces"):
        # Detect faces
        faces = face_cascade.detectMultiScale(face_np, scaleFactor=1.1, minNeighbors=5)

        st.write(f"Detected {len(faces)} face(s).")

        # Overlay emoji on each face
        for (x, y, w, h) in faces:
            face_np = overlay_emoji_no_alpha(face_np, emoji_np, x, y, w, h)

        # Convert back to RGB and display
        result_img = cv2.cvtColor(face_np, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Face Replaced with Emoji", use_column_width=True)
else:
    st.warning("Upload both a face image and an emoji image to proceed.")


# In[ ]:




