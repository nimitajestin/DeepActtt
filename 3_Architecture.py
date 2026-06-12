import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Architecture", page_icon="🏗️", layout="wide")

st.title("Model Architecture")

st.markdown("""
DeepAct uses a **CNN + RNN hybrid architecture** that combines spatial feature extraction with temporal pattern modeling.
""")

# Architecture Diagram
st.header(" Data Flow Diagram")

st.code("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                          │
│  Video Clip (.avi)                                                          │
│  └── OpenCV reads frames                                                    │
│  └── Sample max 20 frames                                                   │
│  └── Each frame: 224 × 224 × 3 (RGB)                                       │
│  Output shape: (batch_size, 20, 224, 224, 3)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              CNN FEATURE EXTRACTOR (InceptionV3)                            │
│  Pretrained on ImageNet (top layer removed)                                 │
│  ├── weights="imagenet"                                                     │
│  ├── include_top=False                                                      │
│  ├── pooling="avg"                                                          │
│  └── input_shape=(224, 224, 3)                                             │
│                                                                             │
│  Per-frame processing:                                                      │
│  Frame_i (224×224×3) → InceptionV3 → 2048-D feature vector                │
│                                                                             │
│  Output: Feature sequence tensor (batch_size, 20, 2048)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRAME MASKING                                        │
│  Boolean mask: (batch_size, 20)                                             │
│  ├── 1 = real frame (not masked)                                           │
│  └── 0 = padded frame (masked)                                             │
│                                                                             │
│  Purpose: Handle variable-length videos without padding noise              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL MODEL (Stacked GRU)                              │
│                                                                             │
│  GRU Layer 1: 16 units, return_sequences=True                              │
│  ├── Input: (batch_size, 20, 2048) + mask                                  │
│  ├── Learns temporal patterns across frames                                │
│  └── Output: (batch_size, 20, 16)                                          │
│                                                                             │
│  GRU Layer 2: 8 units                                                       │
│  ├── Aggregates sequential information                                     │
│  └── Output: (batch_size, 8)                                               │
│                                                                             │
│  Dropout Layer: rate = 0.4                                                  │
│  ├── Prevents overfitting on small dataset                                 │
│  └── Output: (batch_size, 8)                                               │
│                                                                             │
│  Dense Layer: 8 units, ReLU activation                                      │
│  └── Output: (batch_size, 8)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                            │
│  Dense + Softmax                                                            │
│  ├── 5 units (one per action class)                                        │
│  └── Output: Probability distribution (batch_size, 5)                      │
│                                                                             │
│  Classes: CricketShot, PlayingCello, Punch, ShavingBeard, TennisSwing      │
└─────────────────────────────────────────────────────────────────────────────┘
""", language="text")

# Layer Details Table
st.header("Layer-by-Layer Details")

layer_data = {
    "Layer": [
        "Input",
        "CNN Feature Extractor",
        "Frame Mask",
        "GRU Layer 1",
        "GRU Layer 2", 
        "Dropout",
        "Dense (Hidden)",
        "Output"
    ],
    "Type": [
        "Video Input",
        "InceptionV3 (Pretrained)",
        "Boolean Mask",
        "GRU",
        "GRU",
        "Dropout",
        "Dense",
        "Dense + Softmax"
    ],
    "Configuration": [
        "20 frames × 224×224×3",
        "ImageNet weights, top removed, avg pooling",
        "(batch, 20), skip padded timesteps",
        "16 units, return_sequences=True",
        "8 units",
        "Rate = 0.4",
        "8 units, ReLU",
        "5 units, Softmax"
    ],
    "Output Shape": [
        "(batch, 20, 224, 224, 3)",
        "(batch, 20, 2048)",
        "(batch, 20)",
        "(batch, 20, 16)",
        "(batch, 8)",
        "(batch, 8)",
        "(batch, 8)",
        "(batch, 5)"
    ],
    "Parameters": [
        "0",
        "23,851,784 (frozen)",
        "0",
        "98,448",
        "600",
        "0",
        "72",
        "45"
    ]
}

import pandas as pd
layer_df = pd.DataFrame(layer_data)
st.table(layer_df)

# Component Justification
st.header("🔍 Component Justification")

justifications = [
    {
        "component": " InceptionV3 (Pretrained CNN)",
        "why": "Transfer Learning",
        "explanation": "Training a CNN from scratch requires millions of images and significant compute. InceptionV3 pretrained on ImageNet provides rich spatial features (edges, textures, shapes) that generalize well to video frames. The top classification layer is removed so we can extract 2048-D feature vectors."
    },
    {
        "component": " GRU over LSTM",
        "why": "Efficiency",
        "explanation": "GRU has fewer parameters than LSTM (no cell state gate), leading to faster training and less overfitting on small datasets. For short sequences (20 frames), GRU achieves comparable accuracy to LSTM while being computationally efficient."
    },
    {
        "component": " Frame Masking",
        "why": "Variable Length Handling",
        "explanation": "Videos have different lengths. Without masking, padded zero-frames would introduce noise into temporal learning. The boolean mask tells GRU to skip padded timesteps, ensuring only real frames contribute to gradient updates."
    },
    {
        "component": " Dropout (0.4)",
        "why": "Regularization",
        "explanation": "With only 594 training clips, the model can easily memorize the training set. Dropout randomly deactivates 40% of neurons during training, forcing the network to learn robust features and reducing overfitting."
    },
    {
        "component": " Softmax Output",
        "why": "Multi-class Classification",
        "explanation": "Softmax converts raw logits into a probability distribution where all classes sum to 1. This enables confidence analysis and ranked predictions (e.g., CricketShot 39.6%, Punch 24.0%, etc.)."
    }
]

for j in justifications:
    with st.expander(f"{j['component']} — {j['why']}"):
        st.write(j['explanation'])

# Parameter Count Visualization
st.header("Parameter Distribution")

param_data = {
    "Component": ["InceptionV3 (Frozen)", "GRU Layer 1", "GRU Layer 2", "Dense Layers", "Total Trainable"],
    "Parameters": [23851784, 98448, 600, 117, 99165]
}

fig_params = go.Figure(data=[go.Pie(
    labels=param_data["Component"],
    values=param_data["Parameters"],
    hole=0.4,
    marker_colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
)])
fig_params.update_layout(title="Model Parameter Distribution")
st.plotly_chart(fig_params, use_container_width=True)

st.info("""
**Key Insight:** Only ~99K parameters are trainable (GRU + Dense layers). 
The 23.8M InceptionV3 parameters are frozen, making training feasible on limited data.
""")

# Architecture Comparison
st.header(" Architecture Comparison")

comparison_data = {
    "Architecture": ["CNN Only", "RNN Only", "CNN + RNN (Ours)", "3D CNN"],
    "Spatial Features": [" Strong", " Weak", " Strong", " Strong"],
    "Temporal Features": [" None", " Strong", " Strong", " Moderate"],
    "Parameters": ["23.8M", "~100K", "~24M", "~50M+"],
    "Training Time": ["Fast", "Fast", "Moderate", "Slow"],
    "Accuracy": ["Low", "Low", "Moderate", "High"],
    "Data Required": ["Medium", "Low", "Medium", "High"]
}

comp_df = pd.DataFrame(comparison_data)
st.table(comp_df)

st.markdown("""
**Why CNN + RNN?**
- **CNN Only**: Captures spatial features but ignores temporal motion
- **RNN Only**: Can model sequences but lacks visual understanding
- **3D CNN**: Excellent but requires massive data and compute
- **CNN + RNN (Ours)**: Best balance of accuracy, efficiency, and data requirements
""")

# Code Snippet
st.header("Implementation Code")

st.code("""
from tensorflow import keras

# CNN Feature Extractor
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((224, 224, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# Sequence Model (GRU)
def get_sequence_model(label_processor):
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((20, 2048))
    mask_input = keras.Input((20,), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    model = keras.Model([frame_features_input, mask_input], output)
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer="adam", metrics=["accuracy"])
    return model
""", language="python")

st.markdown("---")
st.caption("DeepAct Video Action Recognition | Architecture Page")
