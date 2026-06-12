import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Preprocessing", page_icon="🔬", layout="wide")

st.title(" Preprocessing Pipeline")

st.markdown("""
This page demonstrates the video preprocessing pipeline applied to each video clip before feeding into the model.

**Pipeline Steps:**
1.  **Load** — Read frames via OpenCV
2.  **Crop** — Center-square crop
3.  **Resize** — 224 × 224 pixels
4.  **Convert** — BGR → RGB color space
5.  **Sample** — Max 20 frames per clip
6.  **Mask** — Boolean mask for padding
""")

# Helper functions (standalone for demo)
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# Generate synthetic demo frames for visualization
def generate_demo_frame(size=(480, 640), pattern="gradient"):
    """Generate a synthetic frame for demonstration"""
    if pattern == "gradient":
        frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for i in range(size[0]):
            frame[i, :] = [int(255 * i / size[0]), int(128), int(255 * (1 - i / size[0]))]
        # Add some "action" elements
        cv2.circle(frame, (size[1]//2, size[0]//2), 50, (255, 255, 0), -1)
        cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 255), 3)
    else:
        frame = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return frame

# Step-by-step visualization
st.header(" Step-by-Step Pipeline")

# Step 1: Original Frame
st.subheader("Step 1: Original Frame (BGR from OpenCV)")
original_frame = generate_demo_frame((480, 640))
st.image(original_frame, caption="Original Frame: 640×480 (BGR format)", use_column_width=True)

st.markdown("""
**Details:**
- OpenCV reads video frames in **BGR** format by default
- Original resolution varies per video (typically 320×240 in UCF-101)
- Frame rate: ~25 fps, so a 7-second clip has ~175 frames
""")

# Step 2: Center Crop
st.subheader("Step 2: Center-Square Crop")
cropped_frame = crop_center_square(original_frame)
st.image(cropped_frame, caption=f"Center Crop: {cropped_frame.shape[1]}×{cropped_frame.shape[0]} (square aspect ratio)", use_column_width=True)

st.markdown("""
**Details:**
- Crop to the largest possible square from the center
- Preserves the most important spatial information
- Eliminates black bars or irrelevant edge content
- Formula: `start = (dim // 2) - (min_dim // 2)`
""")

# Step 3: Resize
st.subheader("Step 3: Resize to 224×224")
resized_frame = cv2.resize(cropped_frame, (224, 224))
st.image(resized_frame, caption="Resized: 224×224 (model input size)", use_column_width=True)

st.markdown("""
**Details:**
- Standard ImageNet input size: 224×224 pixels
- Required for InceptionV3 feature extractor
- Bilinear interpolation for smooth resizing
""")

# Step 4: BGR → RGB
st.subheader("Step 4: BGR → RGB Conversion")
rgb_frame = resized_frame[:, :, [2, 1, 0]]
col1, col2 = st.columns(2)
with col1:
    st.image(resized_frame, caption="BGR Format")
with col2:
    st.image(rgb_frame, caption="RGB Format (final input)")

st.markdown("""
**Details:**
- OpenCV uses BGR, but InceptionV3 expects RGB
- Channel reordering: `frame[:, :, [2, 1, 0]]`
- Critical for correct color feature extraction
""")

# Step 5: Frame Sampling
st.header(" Frame Sampling Strategy")

st.markdown("""
Videos have variable lengths, but the model requires fixed input. We use **uniform sampling**:
""")

# Visualize sampling
n_total_frames = st.slider("Total frames in video", 10, 100, 50)
n_sample_frames = 20

sample_indices = np.linspace(0, n_total_frames - 1, min(n_sample_frames, n_total_frames), dtype=int)

# Create visualization
fig_col1, fig_col2 = st.columns([3, 1])
with fig_col1:
    timeline = np.zeros((50, n_total_frames * 10, 3), dtype=np.uint8)
    timeline[:, :] = (200, 200, 200)
    for idx in sample_indices:
        timeline[:, idx*10:(idx+1)*10] = (255, 100, 100)
    st.image(timeline, caption=f"Timeline: Red bars = sampled frames (total: {len(sample_indices)})", use_column_width=True)

with fig_col2:
    st.metric("Total Frames", n_total_frames)
    st.metric("Sampled Frames", len(sample_indices))
    st.metric("Sampling Rate", f"{len(sample_indices)/n_total_frames*100:.1f}%")

st.markdown(f"""
**Sampling Details:**
- Maximum frames per clip: **{n_sample_frames}**
- Sampling method: **Uniform** (evenly spaced)
- For {n_total_frames}-frame video: indices = {sample_indices.tolist()}
- Short clips (< {n_sample_frames} frames): use all frames, pad rest
""")

# Step 6: Masking
st.header(" Frame Masking")

st.markdown("""
Variable-length videos are padded to 20 frames. A **boolean mask** tells the GRU which timesteps are real vs. padded.
""")

# Demo mask visualization
video_length = st.slider("Actual video length (frames)", 1, 20, 12)
mask = np.zeros((1, 20), dtype=bool)
mask[0, :video_length] = True

mask_viz = np.zeros((30, 20 * 30, 3), dtype=np.uint8)
for i in range(20):
    if mask[0, i]:
        mask_viz[:, i*30:(i+1)*30] = (100, 255, 100)  # Green = real
    else:
        mask_viz[:, i*30:(i+1)*30] = (255, 100, 100)  # Red = padded

st.image(mask_viz, caption="Mask Visualization: Green = real frames, Red = padded", use_column_width=True)

st.markdown(f"""
**Mask Details:**
- Mask shape: `(batch_size, 20)`
- Real frames: `mask[i, :length] = 1` (True)
- Padded frames: `mask[i, length:] = 0` (False)
- GRU uses `mask=mask_input` to skip padding timesteps
- Prevents noise from zero-padded frames in temporal learning

**Example for {video_length}-frame video:**
```python
mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
       └────── {video_length} real ──────┘  └─── padded ───┘
```
""")

# Complete Pipeline Summary
st.header(" Complete Pipeline Summary")

st.code("""
def load_video(path, max_frames=20, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = crop_center_square(frame)    # Step 2: Center crop
        frame = cv2.resize(frame, resize)     # Step 3: Resize to 224×224
        frame = frame[:, :, [2, 1, 0]]        # Step 4: BGR → RGB
        frames.append(frame)
        if len(frames) == max_frames:
            break
    cap.release()
    return np.array(frames)                    # Step 5: Return sampled frames

# Feature extraction + masking
frame_features = np.zeros((1, 20, 2048))
frame_mask = np.zeros((1, 20), dtype=bool)
for j in range(min(20, len(frames))):
    frame_features[0, j, :] = feature_extractor.predict(frames[None, j, :])
    frame_mask[0, j] = True                   # Step 6: Set mask
""", language="python")

st.markdown("---")
st.caption("DeepAct Video Action Recognition | Preprocessing Pipeline Page")
