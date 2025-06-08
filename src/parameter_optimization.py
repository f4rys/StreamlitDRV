"""
Parameter optimization and grid analysis for dimensionality reduction methods.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
import umap
from src.metrics import calculate_trustworthiness


def create_heatmap(heatmap_data, x_labels, y_labels, title, colorbar_title, value_format=".3f"):
    """Create a generic heatmap visualization."""
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        colorbar=dict(title=colorbar_title),
        text=[[f"{val:{value_format}}" if not np.isnan(val) else "" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title=title,
        height=400
    )
    
    return fig_heatmap


def analyze_pca_parameters(X_scaled):
    """Analyze PCA parameters through reconstruction error heatmap."""
    st.info("For PCA, we'll analyze reconstruction error across different numbers of components and data subsets.")
    
    # Create parameter grid for PCA
    max_components = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
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
    
    # Create and display heatmap
    fig_heatmap = create_heatmap(
        heatmap_data,
        [f"{ratio:.1f}" for ratio in subset_ratios],
        [str(comp) for comp in component_range],
        "PCA Reconstruction Error Heatmap",
        "Reconstruction Error (MSE)",
        ".6f"
    )
    fig_heatmap.update_layout(
        xaxis_title="Data Subset Ratio",
        yaxis_title="Number of Components"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Find optimal parameters
    min_error_idx = np.nanargmin(reconstruction_errors)
    optimal_params = parameter_combinations[min_error_idx]
    min_error = reconstruction_errors[min_error_idx]
    
    st.success(f"🎯 Optimal parameters: {optimal_params[0]} components with {optimal_params[1]:.1%} of data (Error: {min_error:.6f})")


def analyze_kpca_parameters(X_scaled):
    """Analyze Kernel PCA parameters through quality score heatmap."""
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
                except Exception:
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
        fig_heatmap_kpca = create_heatmap(
            heatmap_data_kpca,
            [str(gamma) for gamma in gamma_values],
            kernel_options,
            "Kernel PCA Quality Score Heatmap",
            "Quality Score (Variance)"
        )
        fig_heatmap_kpca.update_layout(
            xaxis_title="Gamma Parameter",
            yaxis_title="Kernel Type"
        )
        
        st.plotly_chart(fig_heatmap_kpca, use_container_width=True)
        
        # Find optimal parameters
        max_quality_idx = np.nanargmax(quality_scores)
        optimal_params_kpca = param_combinations[max_quality_idx]
        max_quality = quality_scores[max_quality_idx]
        
        st.success(f"🎯 Best parameters: {optimal_params_kpca[0]} kernel with gamma={optimal_params_kpca[1]} (Quality: {max_quality:.3f})")
    else:
        st.error("Could not compute quality scores for any parameter combination.")


def analyze_tsne_parameters(X_scaled):
    """Analyze t-SNE parameters through trustworthiness heatmap."""
    perplexity_values = [5, 15, 30, 50]
    learning_rate_values = [50, 100, 200, 500]
    
    # Use a smaller sample for computational efficiency
    max_samples_nonlinear = min(300, X_scaled.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(X_scaled.shape[0], size=max_samples_nonlinear, replace=False)
    X_sample = X_scaled[sample_indices]
    
    quality_scores = []
    param_combinations = []
    
    with st.spinner("Computing t-SNE quality scores..."):
        for perplexity in perplexity_values:
            for learning_rate in learning_rate_values:
                if perplexity < X_sample.shape[0]:  # Ensure perplexity is valid
                    try:
                        tsne_temp = TSNE(
                            n_components=2, perplexity=perplexity, 
                            learning_rate=learning_rate, random_state=42, 
                            verbose=0, max_iter=300
                        )
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
        
        fig_heatmap = create_heatmap(
            heatmap_data,
            [str(lr) for lr in learning_rate_values],
            [str(perp) for perp in perplexity_values],
            "t-SNE Parameter Quality Heatmap",
            "Trustworthiness Score"
        )
        fig_heatmap.update_layout(
            xaxis_title="Learning Rate",
            yaxis_title="Perplexity"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Find optimal parameters
        max_idx = np.nanargmax(quality_scores)
        optimal_params = param_combinations[max_idx]
        max_score = quality_scores[max_idx]
        
        st.success(f"🎯 Best parameters: Perplexity={optimal_params[0]}, Learning Rate={optimal_params[1]} (Trustworthiness: {max_score:.3f})")


def analyze_umap_parameters(X_scaled):
    """Analyze UMAP parameters through trustworthiness heatmap."""
    n_neighbors_values = [5, 15, 30, 50]
    min_dist_values = [0.01, 0.1, 0.3, 0.5]
    
    # Use a smaller sample for computational efficiency
    max_samples_nonlinear = min(300, X_scaled.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(X_scaled.shape[0], size=max_samples_nonlinear, replace=False)
    X_sample = X_scaled[sample_indices]
    
    quality_scores = []
    param_combinations = []
    
    with st.spinner("Computing UMAP quality scores..."):
        for n_neighbors in n_neighbors_values:
            for min_dist in min_dist_values:
                try:
                    umap_temp = umap.UMAP(
                        n_components=2, n_neighbors=n_neighbors, 
                        min_dist=min_dist, random_state=42
                    )
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
        
        fig_heatmap = create_heatmap(
            heatmap_data,
            [str(md) for md in min_dist_values],
            [str(nn) for nn in n_neighbors_values],
            "UMAP Parameter Quality Heatmap",
            "Trustworthiness Score"
        )
        fig_heatmap.update_layout(
            xaxis_title="Minimum Distance",
            yaxis_title="Number of Neighbors"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Find optimal parameters
        max_idx = np.nanargmax(quality_scores)
        optimal_params = param_combinations[max_idx]
        max_score = quality_scores[max_idx]
        
        st.success(f"🎯 Best parameters: n_neighbors={optimal_params[0]}, min_dist={optimal_params[1]} (Trustworthiness: {max_score:.3f})")


def show_parameter_optimization(X_scaled, method):
    """Show parameter optimization analysis."""
    st.header("6. Reconstruction Error Heatmap for Different Parameters")
    st.markdown("Analyze how different parameter combinations affect reconstruction quality for methods that support it.")
    
    # Only show heatmap for methods that can calculate reconstruction error
    if method in ["PCA", "KPCA"]:
        st.subheader(f"Parameter Grid Analysis for {method}")
        
        if method == "PCA":
            analyze_pca_parameters(X_scaled)
        elif method == "KPCA":
            analyze_kpca_parameters(X_scaled)
    
    elif method in ["t-SNE", "UMAP", "TRIMAP", "PaCMAP"]:
        st.subheader(f"Parameter Grid Analysis for {method}")
        
        # For non-linear methods, we'll use different quality metrics
        st.info(f"For {method}, we'll analyze how different parameters affect the embedding quality using trustworthiness and continuity metrics.")
        
        if method == "t-SNE":
            analyze_tsne_parameters(X_scaled)
        elif method == "UMAP":
            analyze_umap_parameters(X_scaled)
        else:
            st.info(f"Parameter grid analysis for {method} is computationally intensive. Consider implementing with a smaller sample size.")
    
    else:
        st.info("Reconstruction error heatmap is only available for PCA, Kernel PCA, t-SNE, and UMAP methods.")
