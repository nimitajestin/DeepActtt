"""
Inference Utilities
Single video prediction and visualization
"""
import os
import numpy as np
import imageio
from src.preprocessing import load_video, prepare_single_video


def predict_single_video(video_path, model, feature_extractor, label_processor):
    """Predict action class for a single video"""
    class_vocab = label_processor.get_vocabulary()

    # Load and preprocess video
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)

    # Predict
    probabilities = model.predict([frame_features, frame_mask], verbose=0)[0]

    # Sort predictions
    results = []
    for i in np.argsort(probabilities)[::-1]:
        results.append({
            "class": class_vocab[i],
            "probability": float(probabilities[i] * 100)
        })

    predicted_class = results[0]["class"]
    confidence = results[0]["probability"]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": results,
        "frames": frames
    }


def create_gif(frames, output_path="animation.gif", max_frames=20, duration=100):
    """Create GIF from video frames"""
    converted_images = frames[:max_frames].astype(np.uint8)
    imageio.mimsave(output_path, converted_images, duration=duration)
    return output_path


def get_confidence_level(confidence):
    """Get confidence interpretation"""
    if confidence >= 70:
        return "High", "🟢"
    elif confidence >= 40:
        return "Moderate", "🟡"
    else:
        return "Low", "🔴"
