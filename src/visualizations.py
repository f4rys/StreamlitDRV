"""
Visualization components for dimensionality reduction results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import METHOD_INFO


def create_scatter_plot(X_reduced, y, method, target_column, dataset_name):
    """Create an interactive scatter plot of the dimensionality reduction results."""
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
    
    return fig


def show_visualization_results(X_reduced, y, method, target_column, dataset_name):
    """Display the visualization results and method information."""
    st.header(f"4. Visualize 2D {method} Results")
    
    # Create and display scatter plot
    fig = create_scatter_plot(X_reduced, y, method, target_column, dataset_name)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show method description
    if method in METHOD_INFO:
        st.info(METHOD_INFO[method])


def create_variance_plot(explained_var_ratio):
    """Create individual explained variance plot."""
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
    return fig_var


def create_interactive_variance_plot(cumulative_var_ratio, selected_components):
    """Create interactive cumulative variance plot."""
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
    
    return fig_interactive, selected_variance


def create_feature_error_plot(feature_names, feature_errors):
    """Create feature-wise reconstruction error plot."""
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
    return fig_feat_error


def create_rfe_impact_plot(rfe_results):
    """Create feature selection impact plot."""
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
    return fig_rfe
