import streamlit as st

st.set_page_config(
    page_title="DeepAct: Video Action Recognition",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .highlight {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎬 DeepAct</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Neural Networks for Video Action Recognition</p>', unsafe_allow_html=True)

st.markdown("---")

# Project Overview
st.header("📌 Project Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset", "UCF-101", "5-class subset")
with col2:
    st.metric("Architecture", "CNN + RNN", "InceptionV3 + GRU")
with col3:
    st.metric("Accuracy", "~40%", "Top-1 Test")

st.markdown("""
### 🎯 Goal
Accurately recognize human actions in short video clips for applications such as:
- 🎥 **Video Surveillance** — automated threat detection
- 🏃 **Sports Analysis** — action classification in games
- 🛡️ **Content Moderation** — filtering inappropriate actions

### 🧠 Approach
This project combines **spatial feature extraction** using a pretrained CNN (InceptionV3) 
with **temporal pattern modeling** using a stacked GRU network to classify video actions.
""")

# Architecture Summary
st.header("🏗️ Architecture Summary")

st.code("""
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Video Clip (.avi)                                   │
│  └── OpenCV → 20 sampled frames × 224×224×3                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  CNN FEATURE EXTRACTOR: InceptionV3 (ImageNet pretrained)  │
│  └── include_top=False, pooling=avg                         │
│  └── Output: 2048-D feature vector per frame               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  FEATURE SEQUENCE: Tensor shape (batch, 20, 2048)          │
│  + BOOLEAN MASK: (batch, 20) — handles variable lengths    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  TEMPORAL MODEL: Stacked GRU                                │
│  ├── GRU(16, return_sequences=True) — temporal patterns    │
│  ├── GRU(8) — sequence aggregation                          │
│  ├── Dropout(0.4) — regularization                          │
│  ├── Dense(8, ReLU) — hidden layer                          │
│  └── Dense(5, Softmax) — action classification             │
└─────────────────────────────────────────────────────────────┘
""", language="text")

# Key Specifications
st.header("📋 Key Specifications")

spec_col1, spec_col2 = st.columns(2)

with spec_col1:
    st.subheader("Dataset")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | Source | UCF-101 Benchmark |
    | Total Clips | 13,320 (full) / 818 (subset) |
    | Train/Test | 594 / 224 clips |
    | Split Ratio | 72% / 28% |
    | Classes | 5 (CricketShot, PlayingCello, Punch, ShavingBeard, TennisSwing) |
    | Format | .avi video + CSV index |
    """)

with spec_col2:
    st.subheader("Model Hyperparameters")
    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | Image Size | 224 × 224 px |
    | Max Frames | 20 timesteps |
    | CNN Features | 2048-D (InceptionV3) |
    | GRU Units | 16 → 8 |
    | Dropout Rate | 0.4 |
    | Optimizer | Adam (lr=0.001) |
    | Loss | Sparse Categorical Crossentropy |
    | Batch Size | 64 |
    | Epochs | 10 (best-model checkpoint) |
    | Validation Split | 30% of training |
    """)

# Component Justification
st.header("🔍 Component Justification")

justification_data = {
    "Component": [
        "🖼️ InceptionV3 (Pretrained)",
        "⏱️ GRU over LSTM",
        "🎭 Frame Masking",
        "🛡️ Dropout (0.4)",
        "📊 Softmax Output"
    ],
    "Justification": [
        "Transfer learning avoids training from scratch; rich ImageNet features generalize well to video frames",
        "Fewer parameters, faster convergence, comparable accuracy for short temporal sequences (20 frames)",
        "Handles variable-length videos correctly without introducing noise from zero-padded frames",
        "Regularizes the model and reduces overfitting on the relatively small UCF-101 training split",
        "Provides class probability distribution for multi-class action classification across 5 categories"
    ]
}

import pandas as pd
just_df = pd.DataFrame(justification_data)
st.table(just_df)

# Navigation Guide
st.header("🧭 Navigation Guide")

st.info("""
👈 **Use the sidebar to navigate through the project pages:**

1. **📊 Dataset** — Explore the UCF-101 dataset, class distributions, and sample videos
2. **🔬 Preprocessing** — Visualize the video preprocessing pipeline (crop, resize, BGR→RGB)
3. **🏗️ Architecture** — View detailed model architecture, layer configurations, and data flow
4. **🎯 Training** — Train the model with live accuracy/loss curves and overfitting analysis
5. **📈 Evaluation** — View confusion matrix, per-class metrics, and test set performance
6. **🔮 Inference** — Upload your own video and get real-time action predictions
""")

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit | DeepAct Video Action Recognition System")
