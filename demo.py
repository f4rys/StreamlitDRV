import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
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

# --- Sample Size Limitation ---
st.subheader("Sample Size Control")

current_size = X_scaled.shape[0]
st.write(f"Current dataset size: {current_size} samples")

if current_size > 1000:
    st.info("💡 Large datasets may take longer to process. Consider using a sample for faster results.")

use_sample = st.checkbox(
    "Limit sample size for faster processing",
    help="Use a random subset of your data for dimensionality reduction. Useful for large datasets or quick exploration."
)

if use_sample:
    max_samples = min(current_size, 10000)  # Cap at 10k samples
    min_samples = min(100, current_size)  # Don't go below dataset size
    
    # Ensure we have a valid range for the slider
    if min_samples >= max_samples:
        st.warning(f"Dataset is too small ({current_size} samples) for sampling. Using all available data.")
        sample_size = current_size
    else:
        default_sample_size = min(1000, current_size)
        # Ensure default is within valid range
        default_sample_size = max(min_samples, min(default_sample_size, max_samples))
        
        sample_size = st.slider(
            "Number of samples to use:",
            min_value=min_samples,
            max_value=max_samples,
            value=default_sample_size,
            step=max(1, min(100, (max_samples - min_samples) // 10)),  # Dynamic step size
            help=f"Select how many samples to use from the {current_size} available samples"
        )
    
    if sample_size < current_size:
        # Set random seed for reproducibility
        np.random.seed(42)
        sample_indices = np.random.choice(current_size, size=sample_size, replace=False)
        
        X_scaled = X_scaled[sample_indices]
        y = y[sample_indices]
        
        st.success(f"✅ Using random sample of {sample_size} samples (from {current_size} total)")
        st.write(f"Final dataset shape for DR: {X_scaled.shape}")
    else:
        st.write("Using all available samples")
else:
    st.write(f"Using all {current_size} samples")

# --- 3. Choose Method ---
st.header("3. Dimensionality Reduction")

method = st.selectbox(
    "Choose dimensionality reduction method:", ["PCA", "KPCA", "t-SNE", "UMAP", "TRIMAP", "PaCMAP"]
)

# Show performance warning for slow methods on large datasets
if X_scaled.shape[0] > 5000 and method in ["t-SNE", "TRIMAP"]:
    st.warning(f"⚠️ {method} can be slow on large datasets ({X_scaled.shape[0]} samples). Consider using sampling or switching to UMAP/PCA for faster results.")
elif X_scaled.shape[0] > 10000 and method in ["KPCA", "PaCMAP"]:
    st.warning(f"⚠️ {method} may take some time on datasets with {X_scaled.shape[0]} samples. Consider using sampling for faster results.")

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

# --- 6. Metrics and Analysis ---
st.header("5. Dimensionality Reduction Metrics")

# Calculate metrics for methods that support them
if method in ["PCA", "KPCA"]:
    
    # For PCA, we can calculate detailed variance metrics
    if method == "PCA":
        st.subheader("Explained Variance Analysis")
        
        # Create a PCA with more components for analysis
        n_components_analysis = min(len(feature_names), X_scaled.shape[0] - 1)
        pca_full = PCA(n_components=n_components_analysis, random_state=42)
        pca_full.fit(X_scaled)
        
        # Explained variance ratio
        explained_var_ratio = pca_full.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        
        # Create subplot with explained variance
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual explained variance
            fig_var = go.Figure()
            fig_var.add_bar(
                x=list(range(1, len(explained_var_ratio) + 1)),
                y=explained_var_ratio,
                name="Individual Variance",
                text=[f"{val:.3f}" for val in explained_var_ratio],
                textposition="outside"
            )
            fig_var.update_layout(
                title="Explained Variance Ratio by Component",
                xaxis_title="Principal Component",
                yaxis_title="Explained Variance Ratio",
                showlegend=False
            )
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            # Cumulative explained variance
            fig_cum = go.Figure()
            fig_cum.add_scatter(
                x=list(range(1, len(cumulative_var_ratio) + 1)),
                y=cumulative_var_ratio,
                mode='lines+markers',
                name="Cumulative Variance",
                line=dict(color='red', width=3),
                marker=dict(size=8)
            )
            
            # Add horizontal lines for common thresholds
            for threshold in [0.8, 0.9, 0.95]:
                fig_cum.add_hline(
                    y=threshold, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text=f"{threshold*100}%"
                )
            
            fig_cum.update_layout(
                title="Cumulative Explained Variance",
                xaxis_title="Number of Components",
                yaxis_title="Cumulative Variance Ratio",
                showlegend=False
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        
        # Metrics table
        st.subheader("Variance Metrics")
        metrics_data = []
        
        # Find minimum components for different variance thresholds
        thresholds = [0.8, 0.9, 0.95, 0.99]
        for threshold in thresholds:
            min_components = np.argmax(cumulative_var_ratio >= threshold) + 1
            if cumulative_var_ratio[min_components - 1] >= threshold:
                metrics_data.append({
                    "Variance Threshold": f"{threshold*100}%",
                    "Min Components Required": min_components,
                    "Actual Variance Achieved": f"{cumulative_var_ratio[min_components - 1]:.3f}"
                })
        
        # Add current 2D projection metrics
        metrics_data.append({
            "Variance Threshold": "Current 2D",
            "Min Components Required": 2,
            "Actual Variance Achieved": f"{cumulative_var_ratio[1]:.3f}"
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    # Reconstruction Error for PCA and KPCA
    st.subheader("Reconstruction Error")
    
    if method == "PCA":
        # Calculate reconstruction error
        X_reduced_2d = reducer.transform(X_scaled)
        X_reconstructed = reducer.inverse_transform(X_reduced_2d)
        reconstruction_error = mean_squared_error(X_scaled, X_reconstructed)
        
        st.metric("Mean Squared Error (MSE)", f"{reconstruction_error:.6f}")
        st.write("Lower MSE indicates better preservation of original data structure.")
        
        # Feature-wise reconstruction error
        feature_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=0)
        
        fig_feat_error = go.Figure()
        fig_feat_error.add_bar(
            x=feature_names,
            y=feature_errors,
            text=[f"{val:.4f}" for val in feature_errors],
            textposition="outside"
        )
        fig_feat_error.update_layout(
            title="Reconstruction Error by Feature",
            xaxis_title="Features",
            yaxis_title="Mean Squared Error",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_feat_error, use_container_width=True)
    
    elif method == "KPCA":
        st.info("Reconstruction error calculation for Kernel PCA requires additional inverse mapping that may not be directly available.")

# Feature Selection Impact Analysis
if len(feature_names) > 3:  # Only if we have enough features
    st.subheader("Feature Selection Impact")
    
    # RFE analysis
    st.write("Analyzing the impact of feature selection on variance preservation...")
    
    # Create different feature subsets using RFE
    feature_counts = [max(2, len(feature_names) // 4), max(3, len(feature_names) // 2), len(feature_names)]
    feature_counts = sorted(list(set(feature_counts)))  # Remove duplicates and sort
    
    rfe_results = []
    
    for n_features in feature_counts:
        if n_features <= len(feature_names):
            # Use RFE with a simple estimator
            estimator = LinearRegression()
            rfe = RFE(estimator, n_features_to_select=n_features)
            X_rfe = rfe.fit_transform(X_scaled, y)
            
            # Apply PCA to the selected features
            pca_rfe = PCA(n_components=2, random_state=42)
            pca_rfe.fit(X_rfe)
            
            variance_2d = np.sum(pca_rfe.explained_variance_ratio_)
            selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
            
            rfe_results.append({
                "Number of Features": n_features,
                "Selected Features": ", ".join(selected_features[:3] + ["..."] if len(selected_features) > 3 else selected_features),
                "2D Variance Explained": f"{variance_2d:.3f}",
                "Variance Value": variance_2d
            })
    
    # Display results
    rfe_df = pd.DataFrame(rfe_results)
    st.dataframe(rfe_df[["Number of Features", "Selected Features", "2D Variance Explained"]], use_container_width=True)
    
    # Plot feature selection impact
    fig_rfe = go.Figure()
    fig_rfe.add_scatter(
        x=[r["Number of Features"] for r in rfe_results],
        y=[r["Variance Value"] for r in rfe_results],
        mode='lines+markers',
        name="Variance vs Features",
        line=dict(color='green', width=3),
        marker=dict(size=10)
    )
    fig_rfe.update_layout(
        title="Impact of Feature Selection on 2D Variance",
        xaxis_title="Number of Features Selected",
        yaxis_title="2D Explained Variance Ratio",
        showlegend=False
    )
    st.plotly_chart(fig_rfe, use_container_width=True)

else:
    st.subheader("Limited Metrics")
    st.info(f"{method} is a non-linear method. Detailed variance analysis is not directly applicable, but the visualization above shows the method's ability to preserve local/global structure.")

st.sidebar.markdown("---")
st.sidebar.info("StreamlitDRV")
