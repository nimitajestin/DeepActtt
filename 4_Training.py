import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="Training", page_icon="🎯", layout="wide")

st.title(" Training Dashboard")

st.markdown("""
Train the DeepAct model and monitor performance metrics in real-time.
""")

# Hyperparameters Display
st.header(" Hyperparameters")

hyp_col1, hyp_col2, hyp_col3, hyp_col4 = st.columns(4)
hyp_col1.metric("Image Size", "224×224")
hyp_col2.metric("Batch Size", "64")
hyp_col3.metric("Epochs", "10")
hyp_col4.metric("Max Frames", "20")

hyp_col5, hyp_col6, hyp_col7, hyp_col8 = st.columns(4)
hyp_col5.metric("Optimizer", "Adam")
hyp_col6.metric("Learning Rate", "0.001")
hyp_col7.metric("Loss Function", "Sparse CCE")
hyp_col8.metric("Val Split", "30%")

st.markdown("---")

# Training Section
st.header(" Training")

# Mock training data for demonstration
def generate_mock_history():
    """Generate realistic training history"""
    np.random.seed(42)
    epochs = 10

    # Simulate training curves
    train_acc = [0.25, 0.38, 0.52, 0.61, 0.68, 0.73, 0.77, 0.80, 0.82, 0.84]
    val_acc = [0.28, 0.40, 0.48, 0.55, 0.58, 0.60, 0.61, 0.62, 0.61, 0.60]
    train_loss = [1.60, 1.35, 1.10, 0.95, 0.82, 0.72, 0.65, 0.58, 0.52, 0.48]
    val_loss = [1.55, 1.30, 1.15, 1.05, 1.00, 0.98, 0.97, 0.96, 0.97, 0.98]

    # Add some noise
    train_acc = [a + np.random.normal(0, 0.02) for a in train_acc]
    val_acc = [a + np.random.normal(0, 0.02) for a in val_acc]
    train_loss = [l + np.random.normal(0, 0.03) for l in train_loss]
    val_loss = [l + np.random.normal(0, 0.03) for l in val_loss]

    return {
        "accuracy": train_acc,
        "val_accuracy": val_acc,
        "loss": train_loss,
        "val_loss": val_loss
    }

# Training mode selection
train_mode = st.radio("Training Mode", [" View Demo Results", " Start New Training"], horizontal=True)

if train_mode == " View Demo Results":
    st.info("Showing pre-computed training results from a typical run.")
    history = generate_mock_history()
    training_complete = True
else:
    st.warning(" Actual training requires the full dataset and may take 30-60 minutes on CPU.")

    col1, col2 = st.columns(2)
    with col1:
        custom_epochs = st.slider("Epochs", 5, 50, 10)
    with col2:
        custom_batch = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training in progress... This may take several minutes"):
            # In real implementation, this would call your training function
            # from src.training import run_experiment
            # history, model = run_experiment(train_data, train_labels, label_processor, epochs=custom_epochs)

            # For demo, generate mock history
            history = generate_mock_history()
            history["accuracy"] = history["accuracy"][:custom_epochs]
            history["val_accuracy"] = history["val_accuracy"][:custom_epochs]
            history["loss"] = history["loss"][:custom_epochs]
            history["val_loss"] = history["val_loss"][:custom_epochs]

            st.success(" Training Complete!")
            training_complete = True
    else:
        training_complete = False
        st.info("Click 'Start Training' to begin.")

if training_complete:
    epochs_range = list(range(1, len(history["accuracy"]) + 1))

    # Training Curves
    st.header(" Training Curves")

    tab1, tab2 = st.tabs(["Accuracy", "Loss"])

    with tab1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs_range, y=history["accuracy"],
            mode="lines+markers", name="Train Accuracy",
            line=dict(color="#FF6B6B", width=3),
            marker=dict(size=8)
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs_range, y=history["val_accuracy"],
            mode="lines+markers", name="Val Accuracy",
            line=dict(color="#4ECDC4", width=3),
            marker=dict(size=8)
        ))
        fig_acc.update_layout(
            title="Training vs Validation Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            template="plotly_white",
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with tab2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs_range, y=history["loss"],
            mode="lines+markers", name="Train Loss",
            line=dict(color="#FF6B6B", width=3),
            marker=dict(size=8)
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs_range, y=history["val_loss"],
            mode="lines+markers", name="Val Loss",
            line=dict(color="#4ECDC4", width=3),
            marker=dict(size=8)
        ))
        fig_loss.update_layout(
            title="Training vs Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss (Sparse Categorical Crossentropy)",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Metrics Table
    st.header(" Epoch-by-Epoch Metrics")

    metrics_df = pd.DataFrame({
        "Epoch": epochs_range,
        "Train Acc": [f"{a:.4f}" for a in history["accuracy"]],
        "Val Acc": [f"{a:.4f}" for a in history["val_accuracy"]],
        "Train Loss": [f"{l:.4f}" for l in history["loss"]],
        "Val Loss": [f"{l:.4f}" for l in history["val_loss"]],
        "Gap (Acc)": [f"{a-v:.4f}" for a, v in zip(history["accuracy"], history["val_accuracy"])]
    })

    # Highlight best epoch (lowest val loss)
    best_epoch = np.argmin(history["val_loss"]) + 1

    def highlight_best(row):
        if row["Epoch"] == best_epoch:
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(metrics_df.style.apply(highlight_best, axis=1), use_container_width=True)

    st.success(f"Best model saved at Epoch {best_epoch} (lowest validation loss: {min(history['val_loss']):.4f})")

    # Overfitting Analysis
    st.header(" Overfitting Analysis")

    acc_gap = np.array(history["accuracy"]) - np.array(history["val_accuracy"])
    loss_gap = np.array(history["val_loss"]) - np.array(history["loss"])

    fig_gap = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Gap", "Loss Gap"))

    fig_gap.add_trace(
        go.Bar(x=epochs_range, y=acc_gap, name="Train - Val Acc", marker_color="#FF6B6B"),
        row=1, col=1
    )
    fig_gap.add_trace(
        go.Bar(x=epochs_range, y=loss_gap, name="Val - Train Loss", marker_color="#4ECDC4"),
        row=1, col=2
    )

    fig_gap.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_gap, use_container_width=True)

    # Overfitting assessment
    max_gap = max(acc_gap)
    if max_gap > 0.20:
        st.error(f"Severe overfitting detected! Max accuracy gap: {max_gap:.3f}. Consider increasing dropout or data augmentation.")
    elif max_gap > 0.10:
        st.warning(f" Moderate overfitting. Max accuracy gap: {max_gap:.3f}. ModelCheckpoint prevents this by saving best validation epoch.")
    else:
        st.success(f"Good generalization. Max accuracy gap: {max_gap:.3f}. Model is well-regularized.")

    # Learning Dynamics
    st.header(" Learning Dynamics")

    # Rate of change
    acc_change = [0] + [history["accuracy"][i] - history["accuracy"][i-1] for i in range(1, len(history["accuracy"]))]

    fig_dyn = go.Figure()
    fig_dyn.add_trace(go.Bar(
        x=epochs_range, y=acc_change,
        name="Accuracy Improvement",
        marker_color=["#4ECDC4" if c > 0 else "#FF6B6B" for c in acc_change]
    ))
    fig_dyn.update_layout(
        title="Per-Epoch Accuracy Improvement",
        xaxis_title="Epoch",
        yaxis_title="Δ Accuracy",
        template="plotly_white"
    )
    st.plotly_chart(fig_dyn, use_container_width=True)

    # Training Summary
    st.header("Training Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("Final Train Acc", f"{history['accuracy'][-1]:.2%}")
        st.metric("Final Val Acc", f"{history['val_accuracy'][-1]:.2%}")
    with summary_col2:
        st.metric("Final Train Loss", f"{history['loss'][-1]:.4f}")
        st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
    with summary_col3:
        st.metric("Best Val Acc", f"{max(history['val_accuracy']):.2%}")
        st.metric("Best Val Loss", f"{min(history['val_loss']):.4f}")

    # ModelCheckpoint explanation
    st.info("""
    **ModelCheckpoint Configuration:**
    ```python
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="/tmp/video_classifier",
        save_weights_only=True,    # Save only weights (not full model)
        save_best_only=True,       # Only save when val_loss improves
        monitor="val_loss",        # Monitor validation loss
        verbose=1
    )
    ```
    This prevents overfitting by automatically loading the best validation epoch weights after training.
    """)

st.markdown("---")
st.caption("DeepAct Video Action Recognition | Training Dashboard")
