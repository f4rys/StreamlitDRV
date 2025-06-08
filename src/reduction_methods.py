"""
Dimensionality reduction methods and their parameter configurations.
"""
import streamlit as st
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap
import trimap
import pacmap


def show_performance_warning(method, n_samples):
    """Show performance warnings for computationally intensive methods."""
    if n_samples > 5000 and method in ["t-SNE", "TRIMAP"]:
        st.warning(f"⚠️ {method} can be slow on large datasets ({n_samples} samples). Consider using sampling or switching to UMAP/PCA for faster results.")
    elif n_samples > 10000 and method in ["KPCA", "PaCMAP"]:
        st.warning(f"⚠️ {method} may take some time on datasets with {n_samples} samples. Consider using sampling for faster results.")


def create_pca_reducer():
    """Create PCA reducer with default parameters."""
    return PCA(n_components=2, random_state=42)


def create_tsne_reducer():
    """Create t-SNE reducer with configurable parameters."""
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
    return TSNE(n_components=2, perplexity=perplexity, random_state=42)


def create_umap_reducer():
    """Create UMAP reducer with configurable parameters."""
    n_neighbors = st.sidebar.slider("Number of neighbors", 5, 100, 15)
    min_dist = st.sidebar.slider("Minimum distance", 0.0, 1.0, 0.1, 0.05)
    return umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
    )


def create_trimap_reducer():
    """Create TRIMAP reducer with configurable parameters."""
    n_inliers = st.sidebar.slider("Number of inliers", 5, 50, 10)
    n_outliers = st.sidebar.slider("Number of outliers", 1, 20, 5)
    return trimap.TRIMAP(
        n_dims=2, n_inliers=n_inliers, n_outliers=n_outliers, verbose=False
    )


def create_kpca_reducer():
    """Create Kernel PCA reducer with configurable parameters."""
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "poly", "sigmoid", "cosine"], index=0)
    gamma = st.sidebar.slider("Gamma (for rbf/poly/sigmoid)", 0.001, 10.0, 1.0)
    degree = st.sidebar.slider("Degree (for poly)", 2, 5, 3) if kernel == "poly" else 3
    return KernelPCA(
        n_components=2, kernel=kernel, gamma=gamma, degree=degree, random_state=42
    )


def create_pacmap_reducer():
    """Create PaCMAP reducer with configurable parameters."""
    n_neighbors = st.sidebar.slider("Number of neighbors", 5, 100, 10)
    mn_ratio = st.sidebar.slider("MN ratio", 0.1, 1.0, 0.5)
    fp_ratio = st.sidebar.slider("FP ratio", 1.0, 4.0, 2.0)
    return pacmap.PaCMAP(
        n_components=2, n_neighbors=n_neighbors, MN_ratio=mn_ratio, 
        FP_ratio=fp_ratio, random_state=42
    )


def get_reducer(method):
    """Get the appropriate reducer based on the selected method."""
    reducer_map = {
        "PCA": create_pca_reducer,
        "t-SNE": create_tsne_reducer,
        "UMAP": create_umap_reducer,
        "TRIMAP": create_trimap_reducer,
        "KPCA": create_kpca_reducer,
        "PaCMAP": create_pacmap_reducer
    }
    
    return reducer_map[method]()


def apply_dimensionality_reduction(X_scaled, method):
    """Apply dimensionality reduction to the scaled data."""
    reducer = get_reducer(method)
    
    with st.spinner(f"Running {method}..."):
        X_reduced = reducer.fit_transform(X_scaled)
    
    return X_reduced, reducer
