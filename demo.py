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
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from itertools import product
import matplotlib.pyplot as plt

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
            showlegend=False)
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Interactive cumulative variance plot with slider
        st.subheader("Interactive Cumulative Variance Explorer")
        
        # Slider for number of components
        max_components = len(cumulative_var_ratio)
        selected_components = st.slider(
            "Select number of components to analyze:",
            min_value=1,
            max_value=max_components,
            value=min(5, max_components),
            step=1,
            help="Move the slider to see cumulative variance for different numbers of components"
        )
        
        # Create interactive plot
        fig_interactive = go.Figure()
        
        # Add full cumulative variance line (light gray)
        fig_interactive.add_scatter(
            x=list(range(1, len(cumulative_var_ratio) + 1)),
            y=cumulative_var_ratio,
            mode='lines',
            name="Full Range",
            line=dict(color='lightgray', width=2),
            opacity=0.5
        )
        
        # Add selected range (highlighted)
        fig_interactive.add_scatter(
            x=list(range(1, selected_components + 1)),
            y=cumulative_var_ratio[:selected_components],
            mode='lines+markers',
            name=f"Selected ({selected_components} components)",
            line=dict(color='red', width=4),
            marker=dict(size=10, color='red'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        )
        
        # Add vertical line at selected point
        fig_interactive.add_vline(
            x=selected_components,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Components: {selected_components}"
        )
        
        # Add horizontal line at selected variance
        selected_variance = cumulative_var_ratio[selected_components - 1]
        fig_interactive.add_hline(
            y=selected_variance,
            line_dash="dash",
            line_color="blue",
            line_width=3,
            annotation_text=f"Variance: {selected_variance:.3f}"
        )
        
        # Add threshold lines
        for threshold in [0.8, 0.9, 0.95]:
            fig_interactive.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="gray",
                opacity=0.7,
                annotation_text=f"{threshold*100}%"
            )
        
        fig_interactive.update_layout(
            title=f"Interactive Cumulative Variance - {selected_components} Components Explain {selected_variance:.1%} of Variance",
            xaxis_title="Number of Components",
            yaxis_title="Cumulative Variance Ratio",
            height=500,
            showlegend=True,
            legend=dict(x=0.7, y=0.3)
        )
        
        st.plotly_chart(fig_interactive, use_container_width=True)
        
        # Display metrics for selected components
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric(
                "Selected Components",
                selected_components,
                f"{selected_components/max_components:.1%} of total"
            )
        
        with col_metrics2:
            st.metric(
                "Cumulative Variance",
                f"{selected_variance:.3f}",
                f"{selected_variance:.1%}"
            )
        
        with col_metrics3:
            if selected_components < max_components:
                remaining_variance = cumulative_var_ratio[-1] - selected_variance
                st.metric(
                    "Remaining Variance",
                    f"{remaining_variance:.3f}",
                    f"-{remaining_variance:.1%}"
                )
            else:
                st.metric(
                    "Coverage",
                    "Complete",
                    "All variance captured"
                )
        
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

# --- 6. Reconstruction Error Heatmap for Different Parameters ---
st.header("6. Reconstruction Error Heatmap for Different Parameters")
st.markdown("Analyze how different parameter combinations affect reconstruction quality for methods that support it.")

# Only show heatmap for methods that can calculate reconstruction error
if method in ["PCA", "KPCA"]:
    
    # Parameter grid selection
    st.subheader(f"Parameter Grid Analysis for {method}")
    
    if method == "PCA":
        st.info("For PCA, we'll analyze reconstruction error across different numbers of components and data subsets.")
        
        # Create parameter grid for PCA
        max_components = min(10, len(feature_names), X_scaled.shape[0] - 1)
        component_range = range(2, max_components + 1)
        
        # Data subset ratios
        subset_ratios = [0.5, 0.7, 0.8, 0.9, 1.0]
        
        # Calculate reconstruction errors for different parameter combinations
        reconstruction_errors = []
        parameter_combinations = []
        
        with st.spinner("Computing reconstruction errors for parameter grid..."):
            for n_components in component_range:
                for subset_ratio in subset_ratios:
                    # Create subset of data
                    n_samples = int(X_scaled.shape[0] * subset_ratio)
                    if n_samples < n_components:
                        continue
                        
                    np.random.seed(42)
                    subset_indices = np.random.choice(X_scaled.shape[0], size=n_samples, replace=False)
                    X_subset = X_scaled[subset_indices]
                    
                    # Apply PCA
                    pca_temp = PCA(n_components=n_components, random_state=42)
                    X_reduced_temp = pca_temp.fit_transform(X_subset)
                    X_reconstructed_temp = pca_temp.inverse_transform(X_reduced_temp)
                    
                    # Calculate reconstruction error
                    error = mean_squared_error(X_subset, X_reconstructed_temp)
                    
                    reconstruction_errors.append(error)
                    parameter_combinations.append((n_components, subset_ratio))
        
        # Create heatmap data
        heatmap_data = np.full((len(component_range), len(subset_ratios)), np.nan)
        
        for i, (n_comp, subset_ratio) in enumerate(parameter_combinations):
            comp_idx = list(component_range).index(n_comp)
            ratio_idx = subset_ratios.index(subset_ratio)
            heatmap_data[comp_idx, ratio_idx] = reconstruction_errors[i]
        
        # Create heatmap using plotly
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"{ratio:.1f}" for ratio in subset_ratios],
            y=[str(comp) for comp in component_range],
            colorscale='Viridis_r',  # Reverse so lower errors are lighter
            colorbar=dict(title="Reconstruction Error (MSE)"),
            text=[[f"{val:.6f}" if not np.isnan(val) else "" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_heatmap.update_layout(
            title="PCA Reconstruction Error Heatmap",
            xaxis_title="Data Subset Ratio",
            yaxis_title="Number of Components",
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Find optimal parameters
        min_error_idx = np.nanargmin(reconstruction_errors)
        optimal_params = parameter_combinations[min_error_idx]
        min_error = reconstruction_errors[min_error_idx]
        
        st.success(f"🎯 Optimal parameters: {optimal_params[0]} components with {optimal_params[1]:.1%} of data (Error: {min_error:.6f})")
    
    elif method == "KPCA":
        st.info("For Kernel PCA, we'll analyze different kernel parameters and their impact on the embedding quality.")
        
        # Parameter grids for different kernels
        kernel_options = ["rbf", "poly", "sigmoid"]
        gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        # Use a smaller subset for KPCA as it's computationally expensive
        max_samples_kpca = min(500, X_scaled.shape[0])
        np.random.seed(42)
        sample_indices = np.random.choice(X_scaled.shape[0], size=max_samples_kpca, replace=False)
        X_sample = X_scaled[sample_indices]
        
        # For KPCA, we'll measure the quality using explained variance approximation
        quality_scores = []
        param_combinations = []
        
        with st.spinner("Computing quality scores for KPCA parameter grid..."):
            for kernel in kernel_options:
                for gamma in gamma_values:
                    try:
                        # Apply KPCA
                        kpca_temp = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, random_state=42)
                        X_reduced_temp = kpca_temp.fit_transform(X_sample)
                        
                        # Calculate a quality metric (variance of the reduced dimensions)
                        quality = np.sum(np.var(X_reduced_temp, axis=0))
                        
                        quality_scores.append(quality)
                        param_combinations.append((kernel, gamma))
                    except Exception as e:
                        # Skip problematic parameter combinations
                        continue
        
        if quality_scores:
            # Create heatmap data
            heatmap_data_kpca = np.full((len(kernel_options), len(gamma_values)), np.nan)
            
            for i, (kernel, gamma) in enumerate(param_combinations):
                kernel_idx = kernel_options.index(kernel)
                gamma_idx = gamma_values.index(gamma)
                heatmap_data_kpca[kernel_idx, gamma_idx] = quality_scores[i]
            
            # Create heatmap
            fig_heatmap_kpca = go.Figure(data=go.Heatmap(
                z=heatmap_data_kpca,
                x=[str(gamma) for gamma in gamma_values],
                y=kernel_options,
                colorscale='Viridis',
                colorbar=dict(title="Quality Score (Variance)"),
                text=[[f"{val:.3f}" if not np.isnan(val) else "" for val in row] for row in heatmap_data_kpca],
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig_heatmap_kpca.update_layout(
                title="Kernel PCA Quality Score Heatmap",
                xaxis_title="Gamma Parameter",
                yaxis_title="Kernel Type",
                height=400
            )
            
            st.plotly_chart(fig_heatmap_kpca, use_container_width=True)
            
            # Find optimal parameters
            max_quality_idx = np.nanargmax(quality_scores)
            optimal_params_kpca = param_combinations[max_quality_idx]
            max_quality = quality_scores[max_quality_idx]
            
            st.success(f"🎯 Best parameters: {optimal_params_kpca[0]} kernel with gamma={optimal_params_kpca[1]} (Quality: {max_quality:.3f})")
        else:
            st.error("Could not compute quality scores for any parameter combination.")

elif method in ["t-SNE", "UMAP", "TRIMAP", "PaCMAP"]:
    st.subheader(f"Parameter Grid Analysis for {method}")
    
    # For non-linear methods, we'll use different quality metrics
    st.info(f"For {method}, we'll analyze how different parameters affect the embedding quality using trustworthiness and continuity metrics.")
    
    # Import required functions for quality metrics
    from sklearn.metrics import pairwise_distances
    
    def calculate_trustworthiness(X_original, X_embedded, n_neighbors=5):
        """Calculate trustworthiness of the embedding"""
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest neighbors in original space
        nbrs_orig = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_original)
        _, indices_orig = nbrs_orig.kneighbors(X_original)
        
        # Find nearest neighbors in embedded space
        nbrs_emb = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_embedded)
        _, indices_emb = nbrs_emb.kneighbors(X_embedded)
        
        # Calculate trustworthiness
        n = X_original.shape[0]
        trustworthiness = 0
        
        for i in range(n):
            # Get neighbors (excluding self)
            neighbors_orig = set(indices_orig[i][1:])
            neighbors_emb = set(indices_emb[i][1:])
            
            # Count how many embedded neighbors are also original neighbors
            intersection = len(neighbors_orig.intersection(neighbors_emb))
            trustworthiness += intersection / n_neighbors
        
        return trustworthiness / n
    
    # Use a smaller sample for computational efficiency
    max_samples_nonlinear = min(300, X_scaled.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(X_scaled.shape[0], size=max_samples_nonlinear, replace=False)
    X_sample = X_scaled[sample_indices]
    
    if method == "t-SNE":
        perplexity_values = [5, 15, 30, 50]
        learning_rate_values = [50, 100, 200, 500]
        
        quality_scores = []
        param_combinations = []
        
        with st.spinner("Computing t-SNE quality scores..."):
            for perplexity in perplexity_values:
                for learning_rate in learning_rate_values:
                    if perplexity < X_sample.shape[0]:  # Ensure perplexity is valid
                        try:
                            tsne_temp = TSNE(n_components=2, perplexity=perplexity, 
                                           learning_rate=learning_rate, random_state=42, 
                                           verbose=0, max_iter=300)
                            X_embedded = tsne_temp.fit_transform(X_sample)
                            
                            # Calculate trustworthiness
                            trust = calculate_trustworthiness(X_sample, X_embedded)
                            
                            quality_scores.append(trust)
                            param_combinations.append((perplexity, learning_rate))
                        except Exception:
                            continue
        
        if quality_scores:
            # Create heatmap
            heatmap_data = np.full((len(perplexity_values), len(learning_rate_values)), np.nan)
            
            for i, (perp, lr) in enumerate(param_combinations):
                perp_idx = perplexity_values.index(perp)
                lr_idx = learning_rate_values.index(lr)
                heatmap_data[perp_idx, lr_idx] = quality_scores[i]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=[str(lr) for lr in learning_rate_values],
                y=[str(perp) for perp in perplexity_values],
                colorscale='Viridis',
                colorbar=dict(title="Trustworthiness Score"),
                text=[[f"{val:.3f}" if not np.isnan(val) else "" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig_heatmap.update_layout(
                title="t-SNE Parameter Quality Heatmap",
                xaxis_title="Learning Rate",
                yaxis_title="Perplexity",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Find optimal parameters
            max_idx = np.nanargmax(quality_scores)
            optimal_params = param_combinations[max_idx]
            max_score = quality_scores[max_idx]
            
            st.success(f"🎯 Best parameters: Perplexity={optimal_params[0]}, Learning Rate={optimal_params[1]} (Trustworthiness: {max_score:.3f})")
    
    elif method == "UMAP":
        n_neighbors_values = [5, 15, 30, 50]
        min_dist_values = [0.01, 0.1, 0.3, 0.5]
        
        quality_scores = []
        param_combinations = []
        
        with st.spinner("Computing UMAP quality scores..."):
            for n_neighbors in n_neighbors_values:
                for min_dist in min_dist_values:
                    try:
                        umap_temp = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                                            min_dist=min_dist, random_state=42)
                        X_embedded = umap_temp.fit_transform(X_sample)
                        
                        # Calculate trustworthiness
                        trust = calculate_trustworthiness(X_sample, X_embedded)
                        
                        quality_scores.append(trust)
                        param_combinations.append((n_neighbors, min_dist))
                    except Exception:
                        continue
        
        if quality_scores:
            # Create heatmap
            heatmap_data = np.full((len(n_neighbors_values), len(min_dist_values)), np.nan)
            
            for i, (nn, md) in enumerate(param_combinations):
                nn_idx = n_neighbors_values.index(nn)
                md_idx = min_dist_values.index(md)
                heatmap_data[nn_idx, md_idx] = quality_scores[i]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=[str(md) for md in min_dist_values],
                y=[str(nn) for nn in n_neighbors_values],
                colorscale='Viridis',
                colorbar=dict(title="Trustworthiness Score"),
                text=[[f"{val:.3f}" if not np.isnan(val) else "" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig_heatmap.update_layout(
                title="UMAP Parameter Quality Heatmap",
                xaxis_title="Minimum Distance",
                yaxis_title="Number of Neighbors",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Find optimal parameters
            max_idx = np.nanargmax(quality_scores)
            optimal_params = param_combinations[max_idx]
            max_score = quality_scores[max_idx]
            
            st.success(f"🎯 Best parameters: n_neighbors={optimal_params[0]}, min_dist={optimal_params[1]} (Trustworthiness: {max_score:.3f})")
    
    else:
        st.info(f"Parameter grid analysis for {method} is computationally intensive. Consider implementing with a smaller sample size.")

else:
    st.info("Reconstruction error heatmap is only available for PCA, Kernel PCA, t-SNE, and UMAP methods.")

# Performance comparison section
st.subheader("Parameter Selection Guidelines")

guidelines = {
    "PCA": "• Use more components for higher accuracy but lower compression\n• Consider data subset ratio for computational efficiency",
    "KPCA": "• RBF kernel works well for most datasets\n• Higher gamma values capture more local patterns\n• Polynomial kernel good for structured data",
    "t-SNE": "• Lower perplexity for local structure, higher for global\n• Higher learning rate for faster convergence\n• Perplexity should be less than number of samples",
    "UMAP": "• More neighbors preserve global structure\n• Lower min_dist creates tighter clusters\n• Good balance between local and global preservation",
    "TRIMAP": "• More inliers preserve local neighborhoods\n• More outliers help with global structure",
    "PaCMAP": "• Balance MN_ratio and FP_ratio for local vs global preservation"
}

if method in guidelines:
    st.info(guidelines[method])

st.sidebar.markdown("---")
st.sidebar.info("StreamlitDRV")
