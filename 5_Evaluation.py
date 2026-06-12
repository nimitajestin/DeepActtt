import streamlit as st
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="Evaluation", page_icon="📈", layout="wide")

st.title("Model Evaluation")

st.markdown("""
Evaluate model performance on the test set with detailed metrics and visualizations.
""")

# Mock evaluation data
def generate_mock_evaluation():
    np.random.seed(42)
    classes = ["CricketShot", "PlayingCello", "Punch", "ShavingBeard", "TennisSwing"]
    n_test = 224

    # Simulate predictions
    y_true = np.random.choice(range(5), size=n_test, p=[0.22, 0.20, 0.18, 0.20, 0.20])

    # Simulate predictions with some accuracy
    y_pred = []
    for true in y_true:
        if np.random.random() < 0.40:  # 40% accuracy
            y_pred.append(true)
        else:
            y_pred.append(np.random.choice([i for i in range(5) if i != true]))
    y_pred = np.array(y_pred)

    # Confusion matrix
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Overall metrics
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # Sample predictions with probabilities
    sample_preds = []
    for i in range(5):
        probs = np.random.dirichlet(np.ones(5)) * 100
        probs = np.sort(probs)[::-1]
        sample_preds.append({
            "true": classes[y_true[i]],
            "predicted": classes[y_pred[i]],
            "probabilities": {classes[j]: probs[j] for j in range(5)},
            "correct": y_true[i] == y_pred[i]
        })

    return {
        "accuracy": accuracy,
        "loss": 1.45,
        "confusion_matrix": cm,
        "classes": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "sample_predictions": sample_preds
    }

eval_data = generate_mock_evaluation()

# Overall Metrics
st.header(" Overall Test Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Test Accuracy", f"{eval_data['accuracy']:.2%}")
metric_col2.metric("Test Loss", f"{eval_data['loss']:.4f}")
metric_col3.metric("Total Test Clips", "224")
metric_col4.metric("Classes", "5")

st.markdown("---")

# Confusion Matrix
st.header("Confusion Matrix")

fig_cm = ff.create_annotated_heatmap(
    z=eval_data["confusion_matrix"],
    x=eval_data["classes"],
    y=eval_data["classes"],
    colorscale="Blues",
    showscale=True,
    annotation_text=[[str(v) for v in row] for row in eval_data["confusion_matrix"]]
)
fig_cm.update_layout(
    title="Confusion Matrix (Test Set)",
    xaxis_title="Predicted",
    yaxis_title="True",
    height=500
)
st.plotly_chart(fig_cm, use_container_width=True)

# Per-Class Metrics
st.header(" Per-Class Performance")

metrics_table = pd.DataFrame({
    "Class": eval_data["classes"],
    "Precision": [f"{p:.3f}" for p in eval_data["precision"]],
    "Recall": [f"{r:.3f}" for r in eval_data["recall"]],
    "F1-Score": [f"{f:.3f}" for f in eval_data["f1"]],
    "Support": eval_data["support"]
})

# Add macro average
macro_precision = np.mean(eval_data["precision"])
macro_recall = np.mean(eval_data["recall"])
macro_f1 = np.mean(eval_data["f1"])

metrics_table.loc[len(metrics_table)] = [
    "**Macro Avg**",
    f"{macro_precision:.3f}",
    f"{macro_recall:.3f}",
    f"{macro_f1:.3f}",
    sum(eval_data["support"])
]

st.table(metrics_table)

# Per-Class Visualization
st.subheader("Per-Class Metrics Comparison")

fig_perclass = make_subplots(rows=1, cols=3, subplot_titles=("Precision", "Recall", "F1-Score"))

fig_perclass.add_trace(
    go.Bar(x=eval_data["classes"], y=eval_data["precision"], marker_color="#FF6B6B", name="Precision"),
    row=1, col=1
)
fig_perclass.add_trace(
    go.Bar(x=eval_data["classes"], y=eval_data["recall"], marker_color="#4ECDC4", name="Recall"),
    row=1, col=2
)
fig_perclass.add_trace(
    go.Bar(x=eval_data["classes"], y=eval_data["f1"], marker_color="#45B7D1", name="F1"),
    row=1, col=3
)

fig_perclass.update_layout(height=400, showlegend=False)
fig_perclass.update_xaxes(tickangle=45)
st.plotly_chart(fig_perclass, use_container_width=True)

# Sample Predictions
st.header("Sample Predictions")

for i, pred in enumerate(eval_data["sample_predictions"]):
    with st.expander(f"Sample {i+1}: True={pred['true']} | Predicted={pred['predicted']} {'✅' if pred['correct'] else '❌'}"):
        st.write(f"**True Label:** {pred['true']}")
        st.write(f"**Predicted:** {pred['predicted']}")

        # Probability bars
        sorted_probs = sorted(pred["probabilities"].items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            is_top = cls == pred["predicted"]
            is_true = cls == pred["true"]

            label = cls
            if is_top and is_true:
                label += " 🏆"
            elif is_top:
                label += " ⭐"
            elif is_true:
                label += " ✓"

            st.progress(min(prob/100, 1.0), text=f"{label}: {prob:.2f}%")

# Error Analysis
st.header(" Error Analysis")

# Calculate misclassification patterns
misclass = []
for i in range(5):
    for j in range(5):
        if i != j and eval_data["confusion_matrix"][i][j] > 0:
            misclass.append({
                "True": eval_data["classes"][i],
                "Predicted": eval_data["classes"][j],
                "Count": eval_data["confusion_matrix"][i][j]
            })

if misclass:
    misclass_df = pd.DataFrame(misclass).sort_values("Count", ascending=False)
    st.write("**Most Common Misclassifications:**")
    st.dataframe(misclass_df, use_container_width=True)

    # Misclassification heatmap (off-diagonal only)
    off_diag = eval_data["confusion_matrix"].copy()
    np.fill_diagonal(off_diag, 0)

    fig_misclass = ff.create_annotated_heatmap(
        z=off_diag,
        x=eval_data["classes"],
        y=eval_data["classes"],
        colorscale="Reds",
        showscale=True
    )
    fig_misclass.update_layout(title="Misclassification Heatmap (Off-Diagonal)", height=500)
    st.plotly_chart(fig_misclass, use_container_width=True)

# Evaluation Metrics Explanation
st.header("Metrics Explanation")

st.markdown("""
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | Correct / Total | Overall correctness across all classes |
| **Precision** | TP / (TP + FP) | Of predicted positive, how many are correct? |
| **Recall** | TP / (TP + FN) | Of actual positives, how many were found? |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **Support** | Count | Number of samples per class |

**Key Insights:**
- **High precision, low recall**: Model is conservative (few false positives, misses some positives)
- **Low precision, high recall**: Model is aggressive (catches most positives, some false alarms)
- **Balanced F1**: Good trade-off between precision and recall
""")

# Loss Function Details
st.header("Loss Function: Sparse Categorical Crossentropy")

st.latex(r"L = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{y_i})")

st.markdown("""
Where:
- $N$ = number of samples
- $y_i$ = true class label (integer, 0-4)
- $p_{y_i}$ = predicted probability for true class

**Why Sparse Categorical Crossentropy?**
- Labels are integers (not one-hot encoded)
- Efficient for multi-class classification
- Penalizes confident wrong predictions heavily
- Combined with Softmax output for probability distribution
""")

st.markdown("---")
st.caption("DeepAct Video Action Recognition | Evaluation Page")
