"""
Model Architecture
CNN Feature Extractor (InceptionV3) + RNN Sequence Model (GRU)
"""
from tensorflow import keras
from src.config import (
    IMG_SIZE, NUM_FEATURES, MAX_SEQ_LENGTH,
    GRU_UNITS_1, GRU_UNITS_2, DENSE_UNITS, DROPOUT_RATE
)


def build_feature_extractor():
    """Build InceptionV3 feature extractor with ImageNet weights"""
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)

    return keras.Model(inputs, outputs, name="feature_extractor")


def get_sequence_model(label_processor):
    """Build GRU-based sequence model for temporal pattern recognition"""
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # GRU Layer 1: learns temporal patterns across frames
    x = keras.layers.GRU(GRU_UNITS_1, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    # GRU Layer 2: aggregates sequential information
    x = keras.layers.GRU(GRU_UNITS_2)(x)
    # Dropout for regularization
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    # Dense hidden layer
    x = keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    # Output layer with softmax
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)
    rnn_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return rnn_model


def get_model_summary(model):
    """Get model summary as string"""
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)
