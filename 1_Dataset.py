import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")

st.title(" Dataset Exploration")

# Try to load data, use mock data if not available
@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv("/kaggle/input/ucf101-videos/train.csv")
        test_df = pd.read_csv("/kaggle/input/ucf101-videos/test.csv")
        return train_df, test_df, True
    except:
        # Mock data for demonstration
        np.random.seed(42)
        classes = ["CricketShot", "PlayingCello", "Punch", "ShavingBeard", "TennisSwing"]
        train_data = {
            "video_name": [f"v_{c}_g{i}_c{j}.avi" for c in classes for i in range(1, 25) for j in range(1, 6)],
            "tag": [c for c in classes for _ in range(120)]
        }
        test_data = {
            "video_name": [f"v_{c}_g{i}_c{j}.avi" for c in classes for i in range(25, 35) for j in range(1, 6)],
            "tag": [c for c in classes for _ in range(50)]
        }
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        # Trim to match actual sizes
        train_df = train_df.sample(n=594, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=224, random_state=42).reset_index(drop=True)
        return train_df, test_df, False

train_df, test_df, is_real = load_data()

# Dataset Overview
st.header(" Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Clips", len(train_df))
col2.metric("Test Clips", len(test_df))
col3.metric("Total Clips", len(train_df) + len(test_df))
col4.metric("Action Classes", train_df["tag"].nunique())

if not is_real:
    st.warning(" Using simulated data for demonstration. Place actual CSV files at `/kaggle/input/ucf101-videos/` for real data.")

# Train/Test Split Visualization
st.subheader("Train / Test Split")
split_data = pd.DataFrame({
    "Split": ["Training", "Test"],
    "Count": [len(train_df), len(test_df)],
    "Percentage": [round(len(train_df)/(len(train_df)+len(test_df))*100, 1), 
                   round(len(test_df)/(len(train_df)+len(test_df))*100, 1)]
})

fig_split = px.pie(split_data, values="Count", names="Split", 
                   title="Dataset Split Distribution",
                   color_discrete_sequence=["#FF6B6B", "#4ECDC4"],
                   hole=0.4)
fig_split.update_traces(textinfo="percent+label")
st.plotly_chart(fig_split, use_container_width=True)

# Class Distribution
st.header("Class Distribution")

train_dist = train_df["tag"].value_counts().reset_index()
train_dist.columns = ["Class", "Count"]
test_dist = test_df["tag"].value_counts().reset_index()
test_dist.columns = ["Class", "Count"]

fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Training Set", "Test Set"),
                         specs=[[{"type": "bar"}, {"type": "bar"}]])

fig_dist.add_trace(
    go.Bar(x=train_dist["Class"], y=train_dist["Count"], 
           marker_color="#FF6B6B", name="Train"),
    row=1, col=1
)
fig_dist.add_trace(
    go.Bar(x=test_dist["Class"], y=test_dist["Count"], 
           marker_color="#4ECDC4", name="Test"),
    row=1, col=2
)

fig_dist.update_layout(height=400, showlegend=False, title_text="Class Distribution Across Splits")
fig_dist.update_xaxes(tickangle=45)
st.plotly_chart(fig_dist, use_container_width=True)

# Class Balance Analysis
st.subheader("Class Balance Analysis")
balance_data = pd.merge(train_dist, test_dist, on="Class", suffixes=("_Train", "_Test"))
balance_data["Train_Pct"] = (balance_data["Count_Train"] / balance_data["Count_Train"].sum() * 100).round(1)
balance_data["Test_Pct"] = (balance_data["Count_Test"] / balance_data["Count_Test"].sum() * 100).round(1)

fig_balance = go.Figure()
fig_balance.add_trace(go.Bar(name="Train %", x=balance_data["Class"], y=balance_data["Train_Pct"], marker_color="#FF6B6B"))
fig_balance.add_trace(go.Bar(name="Test %", x=balance_data["Class"], y=balance_data["Test_Pct"], marker_color="#4ECDC4"))
fig_balance.update_layout(barmode="group", title="Class Percentage Distribution", xaxis_tickangle=-45)
st.plotly_chart(fig_balance, use_container_width=True)

# Sample Videos
st.header("Sample Videos")

sample_size = st.slider("Number of samples to display", 5, 20, 10)
samples = train_df.sample(sample_size, random_state=42)

st.dataframe(samples, use_container_width=True)

# Dataset Statistics
st.header(" Dataset Statistics")

stats_col1, stats_col2 = st.columns(2)

with stats_col1:
    st.subheader("Training Set")
    st.markdown(f"""
    - **Total clips:** {len(train_df)}
    - **Classes:** {train_df['tag'].nunique()}
    - **Avg clips per class:** {len(train_df) // train_df['tag'].nunique()}
    - **Most common:** {train_dist.iloc[0]['Class']} ({train_dist.iloc[0]['Count']} clips)
    - **Least common:** {train_dist.iloc[-1]['Class']} ({train_dist.iloc[-1]['Count']} clips)
    """)

with stats_col2:
    st.subheader("Test Set")
    st.markdown(f"""
    - **Total clips:** {len(test_df)}
    - **Classes:** {test_df['tag'].nunique()}
    - **Avg clips per class:** {len(test_df) // test_df['tag'].nunique()}
    - **Most common:** {test_dist.iloc[0]['Class']} ({test_dist.iloc[0]['Count']} clips)
    - **Least common:** {test_dist.iloc[-1]['Class']} ({test_dist.iloc[-1]['Count']} clips)
    """)

# UCF-101 Full Dataset Context
st.header(" UCF-101 Full Dataset Context")

st.info("""
**Note:** This project uses a **5-class subset** of the full UCF-101 dataset for demonstration.

| Property | Full UCF-101 | This Project (Subset) |
|----------|-------------|----------------------|
| Total Clips | 13,320 | 818 |
| Action Classes | 101 | 5 |
| Source | YouTube | Kaggle subset |
| Resolution | 320×240 | 224×224 (preprocessed) |
| Duration | ~7 seconds | Variable (max 20 frames) |

The full UCF-101 dataset covers diverse human actions including sports, musical instruments, 
personal care, and fitness activities collected from YouTube videos.
""")

st.markdown("---")
st.caption("DeepAct Video Action Recognition | Dataset Exploration Page")
