"""
Training Utilities
Handles model training with checkpointing and history tracking
"""
import os
from tensorflow import keras
from src.config import EPOCHS, CHECKPOINT_PATH


def run_experiment(train_data, train_labels, label_processor, epochs=EPOCHS):
    """Run training experiment with ModelCheckpoint callback"""
    from src.model import get_sequence_model

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) if os.path.dirname(CHECKPOINT_PATH) else "/tmp", exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )

    seq_model = get_sequence_model(label_processor)
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=1
    )

    # Load best weights
    if os.path.exists(CHECKPOINT_PATH + ".index") or os.path.exists(CHECKPOINT_PATH):
        seq_model.load_weights(CHECKPOINT_PATH)

    return history, seq_model


def evaluate_model(model, test_data, test_labels):
    """Evaluate model on test set"""
    loss, accuracy = model.evaluate(
        [test_data[0], test_data[1]],
        test_labels,
        verbose=0
    )
    return loss, accuracy


def get_training_history(history):
    """Extract metrics from history object"""
    return {
        "accuracy": history.history.get("accuracy", []),
        "val_accuracy": history.history.get("val_accuracy", []),
        "loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", [])
    }
