"""
DeepAct Configuration
Centralized hyperparameters and constants
"""

# Image & Video Processing
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.3

# Model Architecture
GRU_UNITS_1 = 16
GRU_UNITS_2 = 8
DENSE_UNITS = 8
DROPOUT_RATE = 0.4

# Paths
CHECKPOINT_PATH = "/tmp/video_classifier"
DATASET_ROOT = "/kaggle/input/ucf101-videos"
TRAIN_CSV = f"{DATASET_ROOT}/train.csv"
TEST_CSV = f"{DATASET_ROOT}/test.csv"

# Class names (will be loaded from data)
CLASS_NAMES = ["CricketShot", "PlayingCello", "Punch", "ShavingBeard", "TennisSwing"]
