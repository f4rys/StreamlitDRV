import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# Page Configuration
st.set_page_config(layout="wide", page_title="StreamlitDRV - PCA on Diabetes Dataset")

# Title
st.title("StreamlitDRV Demo: PCA on Diabetes Dataset")
st.markdown("""
This simple demo loads the scikit-learn diabetes dataset, performs Principal Component Analysis (PCA)
to reduce it to 2 dimensions, and visualizes the result.
The points are colored by the diabetes progression target variable.
""")

# --- 1. Load Data ---
st.header("1. Load Diabetes Dataset")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

st.subheader("Dataset Preview (First 5 rows)")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape}")

# --- 2. Preprocess Data (Scaling) ---
st.header("2. Data Preprocessing")
st.subheader("Standard Scaling")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.write("Data has been scaled (mean 0, variance 1 for each feature).")

# --- 3. Perform PCA ---
st.header("3. Principal Component Analysis (PCA)")

# Sidebar for PCA n_components (though we'll default to 2 for visualization)
n_components = st.sidebar.slider(
    "Number of Principal Components to compute for explained variance (visualization fixed to 2D)",
    min_value=2,
    max_value=X_scaled.shape[1],
    value=min(5, X_scaled.shape[1]),  # Default to 5 or max features if less
    step=1
)

# Perform PCA
pca = PCA(n_components=n_components)
X_pca_full = pca.fit_transform(X_scaled) # For explained variance calculation

# For 2D visualization, ensure we take only 2 components if n_components was > 2
if n_components == 2:
    X_pca_2d = X_pca_full
else:
    pca_2d_viz = PCA(n_components=2)
    X_pca_2d = pca_2d_viz.fit_transform(X_scaled)


st.subheader("Explained Variance Ratio")
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Create a DataFrame for the explained variance
variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(n_components)],
    'Explained Variance Ratio': explained_variance_ratio,
    'Cumulative Explained Variance': cumulative_explained_variance
})
st.dataframe(variance_df.style.format({
    "Explained Variance Ratio": "{:.2%}",
    "Cumulative Explained Variance": "{:.2%}"
}))

if n_components > 2:
    fig_variance = px.bar(
        variance_df,
        x='Principal Component',
        y='Explained Variance Ratio',
        title='Explained Variance per Principal Component',
        labels={'Explained Variance Ratio': 'Explained Variance Ratio (%)'},
        text_auto=True
    )
    fig_variance.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    st.plotly_chart(fig_variance, use_container_width=True)


# --- 4. Visualize PCA Results (2D) ---
st.header("4. Visualize 2D PCA Results")

# Create a DataFrame for the 2D PCA results for plotting
pca_df = pd.DataFrame(
    data=X_pca_2d,
    columns=['Principal Component 1', 'Principal Component 2']
)
pca_df['target'] = y # Add target variable for coloring

fig = px.scatter(
    pca_df,
    x='Principal Component 1',
    y='Principal Component 2',
    color='target',  # Color points by the diabetes target value
    title='2D PCA of Diabetes Dataset',
    labels={'target': 'Diabetes Progression'},
    hover_data={'target': True} # Show target value on hover
)
fig.update_layout(
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2",
    coloraxis_colorbar_title_text='Diabetes Progression'
)
st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("This is a demo for StreamlitDRV.")

# You can add more details to the sidebar or main page later
st.sidebar.markdown("### Future Steps:")
st.sidebar.markdown("""
- Add more dimensionality reduction techniques (UMAP, t-SNE).
- Implement data uploading.
- Add more metrics for embedding quality.
""")