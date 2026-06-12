# DeepAct: Video Action Recognition System

**Deep Learning Project | UCF-101 Action Recognition Benchmark**

A complete end-to-end video action recognition system using CNN + RNN hybrid architecture.

## Project Overview

| Feature | Details |
|---------|---------|
| **Dataset** | UCF-101 (5-class subset) |
| **Architecture** | InceptionV3 CNN + Stacked GRU |
| **Input** | 20 frames × 224×224×3 per video |
| **Classes** | CricketShot, PlayingCello, Punch, ShavingBeard, TennisSwing |
| **Accuracy** | ~39-40% top-1 (test set) |

## Project Structure

```
deepact_video_recognition/
├── app.py                    # Main entry point
├── pages/                    # Streamlit multi-pages
│   ├── 1_Dataset.py
│   ├── 2_Preprocessing.py
│   ├── 3_Architecture.py
│   ├── 4_Training.py
│   ├── 5_Evaluation.py
│   └── 6_Inference.py
├── src/                      # Core modules (your original code)
│   ├── config.py
│   ├── data_utils.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── training.py
│   └── inference.py
├── assets/                   # Sample videos & images
└── requirements.txt
```

##  Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

##  System Requirements

- Python 3.9+
- 8GB+ RAM recommended
- GPU optional (CPU inference supported)

##  Model Architecture

```
Video (.avi) → OpenCV → [20 frames × 224×224×3]
                                ↓
                     InceptionV3 (ImageNet pretrained)
                                ↓
                    Feature Sequence [batch, 20, 2048]
                           + Boolean Mask [batch, 20]
                                ↓
                          GRU(16, return_sequences=True)
                                ↓
                              GRU(8)
                                ↓
                           Dropout(0.4)
                                ↓
                          Dense(8, ReLU)
                                ↓
                          Dense(5, Softmax)
                                ↓
                    Predicted Action Class
```

##  Citation

UCF-101 Dataset: Soomro et al., "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild", CRCV-TR-12-01, 2012.
