"""
Data loading and preprocessing utilities for StreamlitDRV.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler


def load_sample_dataset():
    """Load the sample diabetes dataset."""
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    return X, y, feature_names, "target", "Diabetes Dataset"


def load_user_dataset():
    """Handle user file upload and data selection."""
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with your data. The app will automatically detect numeric columns for analysis."
    )
    
    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file to continue.")
        st.stop()
    
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
            default=feature_columns[:10] if len(feature_columns) > 10 else feature_columns,
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
        
        return X, y, feature_names, target_column, dataset_name
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()


def handle_missing_values(X, y, feature_names):
    """Handle NaN values in the dataset."""
    nan_count = np.isnan(X).sum()
    total_nans = nan_count.sum()
    
    if total_nans == 0:
        return X, y
    
    st.warning(f"⚠️ Found {total_nans} NaN values in the dataset")
    
    # Show NaN distribution per feature
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
    
    return X, y


def scale_data(X):
    """Scale the input data using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.write("Data has been scaled (mean 0, variance 1 for each feature).")
    return X_scaled, scaler


def handle_sample_size(X_scaled, y):
    """Handle sample size limitation for large datasets."""
    st.subheader("Sample Size Control")
    
    current_size = X_scaled.shape[0]
    st.write(f"Current dataset size: {current_size} samples")
    
    if current_size > 1000:
        st.info("💡 Large datasets may take longer to process. Consider using a sample for faster results.")
    
    use_sample = st.checkbox(
        "Limit sample size for faster processing",
        help="Use a random subset of your data for dimensionality reduction. Useful for large datasets or quick exploration."
    )
    
    if not use_sample:
        st.write(f"Using all {current_size} samples")
        return X_scaled, y
    
    max_samples = min(current_size, 10000)  # Cap at 10k samples
    min_samples = min(100, current_size)  # Don't go below dataset size
    
    # Ensure we have a valid range for the slider
    if min_samples >= max_samples:
        st.warning(f"Dataset is too small ({current_size} samples) for sampling. Using all available data.")
        return X_scaled, y
    
    default_sample_size = min(1000, current_size)
    # Ensure default is within valid range
    default_sample_size = max(min_samples, min(default_sample_size, max_samples))
    
    sample_size = st.slider(
        "Number of samples to use:",
        min_value=min_samples,
        max_value=max_samples,
        value=default_sample_size,
        step=max(1, min(100, (max_samples - min_samples) // 10)),
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
    
    return X_scaled, y


def show_dataset_preview(X, y, feature_names, target_column, dataset_name, data_source):
    """Display dataset preview and basic information."""
    st.subheader("Dataset Preview (First 5 rows)")
    
    # Create a preview dataframe with the current features and target
    if data_source == "Upload Your Own Data":
        preview_df = pd.DataFrame(X, columns=feature_names)
        preview_df[target_column] = y
    else:
        preview_df = pd.DataFrame(X, columns=feature_names)
        preview_df[target_column] = y
    
    st.dataframe(preview_df.head())
    st.write(f"Dataset shape: {X.shape}")
    st.write(f"Selected features: {len(feature_names)} columns")
    st.write(f"Target column: {target_column}")
