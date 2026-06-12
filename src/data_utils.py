"""
Data Loading Utilities
Handles CSV reading, dataset statistics, and label encoding
"""
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from src.config import TRAIN_CSV, TEST_CSV, CLASS_NAMES


def load_dataset():
    """Load train and test CSV files"""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, test_df


def get_dataset_stats(train_df, test_df):
    """Calculate dataset statistics"""
    total_train = len(train_df)
    total_test = len(test_df)
    total = total_train + total_test

    stats = {
        "total_train": total_train,
        "total_test": total_test,
        "total": total,
        "train_pct": round(total_train / total * 100, 1),
        "test_pct": round(total_test / total * 100, 1),
        "num_classes": train_df["tag"].nunique(),
        "class_distribution": train_df["tag"].value_counts().to_dict(),
        "avg_clips_per_class": round(total_train / train_df["tag"].nunique(), 1)
    }
    return stats


def create_label_processor(train_df):
    """Create StringLookup layer for label encoding"""
    label_processor = keras.layers.StringLookup(
        num_oov_indices=0,
        vocabulary=np.unique(train_df["tag"])
    )
    return label_processor


def get_class_names(label_processor):
    """Get vocabulary/class names from label processor"""
    return label_processor.get_vocabulary()


def get_sample_videos(df, n=5, random_state=42):
    """Get random sample videos for display"""
    return df.sample(n, random_state=random_state)
