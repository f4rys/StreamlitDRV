"""
Configuration and imports for StreamlitDRV application.
"""

import streamlit as st


# Page Configuration
def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        layout="wide", 
        page_title="StreamlitDRV - Dimensionality Reduction Visualization",
    )


def show_header():
    """Display application header and description."""
    st.title("StreamlitDRV: Dimensionality Reduction Visualization")
    st.markdown(
        """
    This app performs dimensionality reduction to reduce data to 2 dimensions and visualizes the result.
    You can use the built-in diabetes dataset or upload your own CSV/Excel file.
    """
    )


# Method information and guidelines
METHOD_INFO = {
    "PCA": "Linear dimensionality reduction that finds principal components with maximum variance.",
    "KPCA": "Kernel PCA uses kernel methods to perform non-linear dimensionality reduction.",
    "t-SNE": "Non-linear technique that preserves local neighborhood structure.",
    "UMAP": "Non-linear technique that preserves both local and global structure.",
    "TRIMAP": "Preserves local neighborhoods while respecting global distances.",
    "PaCMAP": "Pairwise Controlled Manifold Approximation Projection preserves local and global structure.",
}


PARAMETER_GUIDELINES = {
    "PCA": "• Use more components for higher accuracy but lower compression\n• Consider data subset ratio for computational efficiency",
    "KPCA": "• RBF kernel works well for most datasets\n• Higher gamma values capture more local patterns\n• Polynomial kernel good for structured data",
    "t-SNE": "• Lower perplexity for local structure, higher for global\n• Higher learning rate for faster convergence\n• Perplexity should be less than number of samples",
    "UMAP": "• More neighbors preserve global structure\n• Lower min_dist creates tighter clusters\n• Good balance between local and global preservation",
    "TRIMAP": "• More inliers preserve local neighborhoods\n• More outliers help with global structure",
    "PaCMAP": "• Balance MN_ratio and FP_ratio for local vs global preservation"
}
