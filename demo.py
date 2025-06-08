import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import plotly.express as px
import umap
import trimap
import pacmap

# Page Configuration
st.set_page_config(
    layout="wide", page_title="StreamlitDRV - Dimensionality Reduction Visualization",
)

# Title
st.title("StreamlitDRV: Dimensionality Reduction Visualization")
st.markdown(
    """
This app performs dimensionality reduction to reduce data to 2 dimensions and visualizes the result.
You can use the built-in diabetes dataset or upload your own CSV/Excel file.
"""
)

# --- 1. Data Source Selection ---
st.header("1. Select Data Source")

data_source = st.radio(
    "Choose your data source:",
    ["Sample Dataset (Diabetes)", "Upload Your Own Data"]
)

if data_source == "Sample Dataset (Diabetes)":
    st.subheader("Loading Diabetes Dataset")
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    target_column = "target"
    dataset_name = "Diabetes Dataset"
    
else:
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with your data. The app will automatically detect numeric columns for analysis."
    )
    
    if uploaded_file is not None:
        try:
            # Read the file based on its extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Successfully loaded {uploaded_file.name}")
            
            # Let user select target column
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("Your dataset needs at least 2 numeric columns for dimensionality reduction.")
                st.stop()
            
            target_column = st.selectbox(
                "Select target column (for coloring the visualization):",
                options=numeric_columns,
                help="This column will be used to color the points in the visualization"
            )
            
            # Select feature columns (all numeric except target)
            feature_columns = [col for col in numeric_columns if col != target_column]
            
            if len(feature_columns) < 2:
                st.error("You need at least 2 feature columns (excluding target) for dimensionality reduction.")
                st.stop()
                
            selected_features = st.multiselect(
                "Select feature columns to use for dimensionality reduction:",
                options=feature_columns,
                default=feature_columns[:10] if len(feature_columns) > 10 else feature_columns,  # Limit to first 10 by default
                help="Select the numeric columns to use as features for dimensionality reduction"
            )
            
            if len(selected_features) < 2:
                st.error("Please select at least 2 feature columns.")
                st.stop()
            
            # Prepare data
            X = df[selected_features].values
            y = df[target_column].values
            feature_names = selected_features
            dataset_name = uploaded_file.name
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    else:
        st.info("Please upload a CSV or Excel file to continue.")
        st.stop()

st.subheader("Dataset Preview (First 5 rows)")
# Create a preview dataframe with the current features and target
if data_source == "Upload Your Own Data":
    preview_df = pd.DataFrame(X, columns=feature_names)
    preview_df[target_column] = y
else:
    preview_df = df

st.dataframe(preview_df.head())
st.write(f"Dataset shape: {X.shape}")
st.write(f"Selected features: {len(feature_names)} columns")
st.write(f"Target column: {target_column}")

# --- 2. Preprocess Data ---
st.header("2. Data Preprocessing")

# Check for NaN values
nan_count = np.isnan(X).sum()
total_nans = nan_count.sum()

if total_nans > 0:
    st.warning(f"⚠️ Found {total_nans} NaN values in the dataset")
    
    # Show NaN distribution per feature
    if data_source == "Upload Your Own Data":
        nan_info = pd.DataFrame({
            'Feature': feature_names,
            'NaN Count': nan_count,
            'NaN Percentage': (nan_count / len(X)) * 100
        })
        nan_info = nan_info[nan_info['NaN Count'] > 0]
        if not nan_info.empty:
            st.write("NaN distribution per feature:")
            st.dataframe(nan_info)
    
    handle_nans = st.radio(
        "How would you like to handle NaN values?",
        ["Drop rows with NaN values", "Stop processing (fix data first)"],
        help="Dimensionality reduction algorithms cannot handle NaN values"
    )
    
    if handle_nans == "Stop processing (fix data first)":
        st.error("Please clean your data and remove NaN values before proceeding.")
        st.stop()
    else:
        # Drop rows with NaN values
        before_shape = X.shape
        valid_indices = ~np.isnan(X).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        st.success(f"✅ Dropped {before_shape[0] - X.shape[0]} rows with NaN values")
        st.write(f"Dataset shape after cleaning: {X.shape} (was {before_shape})")
        
        if X.shape[0] < 10:
            st.error("Too few samples remaining after dropping NaN values. Please clean your data.")
            st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.write("Data has been scaled (mean 0, variance 1 for each feature).")

# --- 3. Choose Method ---
st.header("3. Dimensionality Reduction")

method = st.selectbox(
    "Choose dimensionality reduction method:", ["PCA", "KPCA", "t-SNE", "UMAP", "TRIMAP", "PaCMAP"]
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

elif method == "KPCA":
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "poly", "sigmoid", "cosine"], index=0)
    gamma = st.sidebar.slider("Gamma (for rbf/poly/sigmoid)", 0.001, 10.0, 1.0)
    degree = st.sidebar.slider("Degree (for poly)", 2, 5, 3) if kernel == "poly" else 3
    reducer = KernelPCA(
        n_components=2, kernel=kernel, gamma=gamma, degree=degree, random_state=42
    )

elif method == "PaCMAP":
    n_neighbors = st.sidebar.slider("Number of neighbors", 5, 100, 10)
    mn_ratio = st.sidebar.slider("MN ratio", 0.1, 1.0, 0.5)
    fp_ratio = st.sidebar.slider("FP ratio", 1.0, 4.0, 2.0)
    reducer = pacmap.PaCMAP(
        n_components=2, n_neighbors=n_neighbors, MN_ratio=mn_ratio, 
        FP_ratio=fp_ratio, random_state=42
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
reduced_df[target_column] = y

# Create scatter plot
fig = px.scatter(
    reduced_df,
    x=f"{method} Component 1",
    y=f"{method} Component 2",
    color=target_column,
    title=f"2D {method} of {dataset_name}",
    labels={target_column: target_column.replace('_', ' ').title()},
    hover_data={target_column: True},
)
fig.update_layout(coloraxis_colorbar_title_text=target_column.replace('_', ' ').title())
st.plotly_chart(fig, use_container_width=True)

# Method descriptions
method_info = {
    "PCA": "Linear dimensionality reduction that finds principal components with maximum variance.",
    "KPCA": "Kernel PCA uses kernel methods to perform non-linear dimensionality reduction.",
    "t-SNE": "Non-linear technique that preserves local neighborhood structure.",
    "UMAP": "Non-linear technique that preserves both local and global structure.",
    "TRIMAP": "Preserves local neighborhoods while respecting global distances.",
    "PaCMAP": "Pairwise Controlled Manifold Approximation Projection preserves local and global structure.",
}

st.info(method_info[method])

st.sidebar.markdown("---")
st.sidebar.info("StreamlitDRV")
