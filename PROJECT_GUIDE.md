# DeepAct: Video Action Recognition

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py

# 3. Open browser at http://localhost:8501
```

## Project Structure

```
deepact_video_recognition/
├── app.py                    ← Main entry point
├── pages/                    ← Streamlit multi-pages
│   ├── 1_Dataset.py       ← Dataset exploration
│   ├── 2_Preprocessing.py ← Preprocessing pipeline demo
│   ├── 3_Architecture.py  ← Model architecture details
│   ├── 4_Training.py      ← Training dashboard
│   ├── 5_Evaluation.py    ← Evaluation metrics
│   └── 6_Inference.py     ← Video prediction
├── src/                      ← Core modules (YOUR ORIGINAL CODE)
│   ├── __init__.py
│   ├── config.py             ← Centralized hyperparameters
│   ├── data_utils.py         ← Data loading utilities
│   ├── preprocessing.py      ← Video preprocessing (unchanged)
│   ├── model.py              ← CNN+GRU architecture (unchanged)
│   ├── training.py           ← Training loop (unchanged)
│   └── inference.py          ← Prediction utilities (unchanged)
├── assets/                   ← Sample videos and images
├── .streamlit/
│   └── config.toml           ← Streamlit configuration
├── requirements.txt
└── README.md
```

## What Was Added (Your Original Code is Unchanged)

| Feature | Description |
|---------|-------------|
| **Multi-page Streamlit app** | Professional navigation with 6 pages |
| **Dataset Explorer** | Interactive class distribution, sample viewing |
| **Preprocessing Visualization** | Step-by-step pipeline demo with synthetic frames |
| **Architecture Diagrams** | ASCII data flow, layer tables, component justifications |
| **Training Dashboard** | Live curves, overfitting analysis, epoch metrics table |
| **Evaluation Suite** | Confusion matrix, per-class metrics, error analysis |
| **Inference UI** | Video upload, probability visualization, confidence analysis |
| **Centralized Config** | Single source of truth for all hyperparameters |
| **Theming** | Custom color scheme and layout |

## How to Use with Real Data

1. Place your UCF-101 CSV files at:
   - `/kaggle/input/ucf101-videos/train.csv`
   - `/kaggle/input/ucf101-videos/test.csv`

2. Update `src/config.py` paths if needed:
   ```python
   DATASET_ROOT = "/kaggle/input/ucf101-videos"
   ```

3. In each page, uncomment the real data loading:
   ```python
   # from src.data_utils import load_dataset
   # train_df, test_df = load_dataset()
   ```

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'src'`
**Fix:** Run from project root: `streamlit run app.py`

**Issue:** Pages not showing in sidebar
**Fix:** Ensure `pages/` folder is in same directory as `app.py`

**Issue:** TensorFlow GPU not working
**Fix:** Install tensorflow-gpu or use CPU (code works on both)

## Customization

- **Colors:** Edit `.streamlit/config.toml`
- **Classes:** Update `src/config.py` CLASS_NAMES
- **Hyperparameters:** Modify `src/config.py` values
- **Layout:** Adjust column ratios in page files
