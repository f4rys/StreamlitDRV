import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import umap
import trimap

# Page Configuration
st.set_page_config(
    layout="wide", page_title="StreamlitDRV - Dimensionality Reduction Demo"
)

# Title
st.title("StreamlitDRV Demo: Dimensionality Reduction on Diabetes Dataset")
st.markdown(
    """
This demo loads the scikit-learn diabetes dataset, performs dimensionality reduction
to reduce it to 2 dimensions, and visualizes the result.
"""
)

# --- 1. Load Data ---
st.header("1. Load Diabetes Dataset")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

st.subheader("Dataset Preview (First 5 rows)")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape}")

# --- 2. Preprocess Data ---
st.header("2. Data Preprocessing")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.write("Data has been scaled (mean 0, variance 1 for each feature).")

# --- 3. Choose Method ---
st.header("3. Dimensionality Reduction")

method = st.selectbox(
    "Choose dimensionality reduction method:", ["PCA", "t-SNE", "UMAP", "TRIMAP"]
)

# Method-specific parameters
st.sidebar.header("Parameters")

if method == "PCA":
    # No additional parameters needed for 2D visualization
    reducer = PCA(n_components=2, random_state=42)

elif method == "t-SNE":
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)

elif method == "UMAP":
    n_neighbors = st.sidebar.slider("Number of neighbors", 5, 100, 15)
    min_dist = st.sidebar.slider("Minimum distance", 0.0, 1.0, 0.1, 0.05)
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
    )

elif method == "TRIMAP":
    n_inliers = st.sidebar.slider("Number of inliers", 5, 50, 10)
    n_outliers = st.sidebar.slider("Number of outliers", 1, 20, 5)
    reducer = trimap.TRIMAP(
        n_dims=2, n_inliers=n_inliers, n_outliers=n_outliers, verbose=False
    )

# --- 4. Apply Reduction ---
with st.spinner(f"Running {method}..."):
    X_reduced = reducer.fit_transform(X_scaled)

# --- 5. Visualize Results ---
st.header(f"4. Visualize 2D {method} Results")

# Create DataFrame for plotting
reduced_df = pd.DataFrame(
    data=X_reduced, columns=[f"{method} Component 1", f"{method} Component 2"]
)
reduced_df["target"] = y

# Create scatter plot
fig = px.scatter(
    reduced_df,
    x=f"{method} Component 1",
    y=f"{method} Component 2",
    color="target",
    title=f"2D {method} of Diabetes Dataset",
    labels={"target": "Diabetes Progression"},
    hover_data={"target": True},
)
fig.update_layout(coloraxis_colorbar_title_text="Diabetes Progression")
st.plotly_chart(fig, use_container_width=True)

# Method descriptions
method_info = {
    "PCA": "Linear dimensionality reduction that finds principal components with maximum variance.",
    "t-SNE": "Non-linear technique that preserves local neighborhood structure.",
    "UMAP": "Non-linear technique that preserves both local and global structure.",
    "TRIMAP": "Preserves local neighborhoods while respecting global distances.",
}

st.info(method_info[method])

st.sidebar.markdown("---")
st.sidebar.info("StreamlitDRV Demo")
