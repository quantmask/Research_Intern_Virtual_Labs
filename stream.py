import os
import scipy.io
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import streamlit as st
import pywt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import Binarizer
from sklearn.manifold import TSNE

# Define output folder for CSV files
output_folder = "dataset_csv"
os.makedirs(output_folder, exist_ok=True)

# Streamlit UI components
st.set_page_config(page_title="Data Processing and Feature Extraction for Fault Analysis", page_icon="📊", layout="wide")
st.title("Data Processing and Feature Extraction for Fault Analysis")
st.write("This application is tuned for processing several data.")

# Sidebar for user input
st.sidebar.header("Analysis Parameters")
max_rows_input = st.sidebar.text_input("Total number of data entry points", value="120000")
plot_duration_input = st.sidebar.text_input("Plot Duration (seconds)", value="1")
sampling_rate_input = st.sidebar.text_input("Sampling Rate (samples per second)", value="12000")

# Feature extraction parameters
feature_options = {
    'Mean': 'Calculate mean value',
    'Standard Deviation': 'Calculate standard deviation',
    'Variance': 'Calculate variance',
    'Kurtosis': 'Calculate kurtosis',
    'Skewness': 'Calculate skewness',
    'Peak Value': 'Calculate peak value',
    'Range': 'Calculate range',
    'RMS': 'Calculate Root Mean Square',
    'Impulse Factor': 'Calculate impulse factor',
    'Crest Factor': 'Calculate crest factor',
    'Shape Factor': 'Calculate shape factor',
    'L1 Normal': 'Calculate L1 normal',
    'L2 Normal': 'Calculate L2 normal'
}

# Fetch available CSV files from the dataset_csv folder
csv_files = [file for file in os.listdir(output_folder) if file.endswith('.csv')]

# Sidebar UI for selecting files
st.sidebar.subheader("File Selection for Feature Extraction")
select_all_files = st.sidebar.checkbox("Select All Files", value=False)

if select_all_files:
    selected_files = csv_files  # Automatically select all files if "Select All Files" is checked
else:
    selected_files = st.sidebar.multiselect("Select Files for Feature Extraction", options=csv_files)
# File uploader for .mat files
uploaded_files = st.file_uploader("Upload .mat files", type=["mat"], accept_multiple_files=True)

if uploaded_files:
    # Extract keys from all uploaded files
    st.subheader("Select Data Keys for Extraction")
    file_keys_mapping = {}
    user_selection = {}

    for mat_file in uploaded_files:
        mat_data = scipy.io.loadmat(mat_file)
        keys = [key for key in mat_data.keys() if not key.startswith("__")]
        file_keys_mapping[mat_file.name] = keys

        # Display keys and allow user selection
        st.write(f"Keys for '{mat_file.name}':")
        selected_keys = st.multiselect(f"Select keys to extract from '{mat_file.name}'", options=keys, key=f"{mat_file.name}_keys")
        user_selection[mat_file.name] = selected_keys

    # Button to confirm selection of data keys
    if st.button("Confirm Data Key Selection"):
        if any(user_selection.values()):
            st.success("Data keys selected. You can now process the .mat files.")
        else:
            st.warning("Please select at least one data key before proceeding.")

    # Button to process the selected keys
    if st.button("Process MAT Files"):
        all_dataframes = []
        all_csv_names = []

        for mat_file in uploaded_files:
            file_name = mat_file.name
            if file_name in user_selection and user_selection[file_name]:
                selected_keys = user_selection[file_name]
                mat_data = scipy.io.loadmat(mat_file)

                for data_key in selected_keys:
                    data = mat_data[data_key]
                    df = pd.DataFrame(data)

                    # Convert max_rows_input to an integer, with error handling
                    try:
                        max_rows = int(max_rows_input) if max_rows_input.isdigit() else len(df)
                    except ValueError:
                        st.error("Please enter a valid integer for maximum rows.")
                        max_rows = len(df)  # Fallback to the length of the DataFrame

                    # Limit rows to user-defined maximum
                    df = df.iloc[:max_rows]

                    # Create CSV filename and save
                    csv_name = f"{os.path.splitext(file_name)[0]}_{data_key}.csv"
                    csv_filename = os.path.join(output_folder, csv_name)  # Ensure output_folder is defined
                    df.to_csv(csv_filename, index=False)

                    # Display the extracted data
                    st.subheader(f"Data from '{data_key}' in '{file_name}':")
                    st.dataframe(df)

                    st.success(f"File '{csv_name}' successfully created with {df.shape[0]} rows.")

        st.success("Selected keys have been processed and saved as CSV files.")

def clean_and_save_dataframe(df, csv_name, max_rows):
    """Clean and save DataFrame with its original name."""
    if df.shape[0] > max_rows:
        df_cleaned = df.iloc[:max_rows]
        csv_filename = os.path.join(output_folder, csv_name)
        df_cleaned.to_csv(csv_filename, index=False)
        st.success(f"File '{csv_filename}' successfully created with {df_cleaned.shape[0]} rows.")
    else:
        st.warning(f"DataFrame '{csv_name}' has only {df.shape[0]} rows; no rows dropped.")

def calculate_features(window):
    """Calculate features for a given window of data."""
    features = {
        'Mean': np.mean(window),
        'Standard Deviation': np.std(window),
        'Variance': np.var(window),
        'Kurtosis': kurtosis(window, fisher=True),
        'Skewness': skew(window),
        'Peak Value': np.max(np.abs(window)),
        'Range': np.max(window) - np.min(window),
        'RMS': np.sqrt(np.mean(np.square(window))),
        'Impulse Factor': np.max(np.abs(window)) / np.mean(np.abs(window)) if np.mean(np.abs(window)) != 0 else np.nan,
        'Crest Factor': np.max(np.abs(window)) / np.sqrt(np.mean(np.square(window))) if np.sqrt(np.mean(np.square(window))) != 0 else np.nan,
        'Shape Factor': np.sqrt(np.mean(np.square(window))) / np.mean(np.abs(window)) if np.mean(np.abs(window)) != 0 else np.nan,
        'L1 Normal': np.sum(np.abs(window)),
        'L2 Normal': np.sqrt(np.sum(np.square(window)))
    }
    return features

def extract_features_from_dataframe(dataframe, window_size):
    """Extract specified features from the DataFrame using a sliding window."""
    features_list = []
    
    for column in dataframe.columns:
        values = dataframe[column].values
        
        # Sliding window feature extraction
        for start in range(0, len(values) - window_size + 1):
            window = values[start:start + window_size]
            features = calculate_features(window)
            features_list.append({**features, **{'Column': column}})
    
    return pd.DataFrame(features_list)

def extract_features_for_all_data_points(dataframe):
    """Extract specified features from the DataFrame."""
    features_dict = {}
    
    for column in dataframe.columns:
        values = dataframe[column]

        # Calculate metrics
        mean_value = np.mean(values)
        std_value = np.std(values)
        kurtosis_value = kurtosis(values, fisher=True)
        peak_value = np.max(values)
        peak_to_peak_value = peak_value - np.min(values)
        impulse_factor = peak_value / mean_value if mean_value != 0 else np.nan

        # Additional metrics
        skewness_value = skew(values)  # Skewness
        crest_factor = peak_value / np.sqrt(np.mean(np.square(values))) if np.sqrt(np.mean(np.square(values))) != 0 else np.nan  # Crest Factor
        shape_factor = np.sqrt(np.mean(np.square(values))) / mean_value if mean_value != 0 else np.nan  # Shape Factor
        rms_value = np.sqrt(np.mean(np.square(values)))  # RMS
        range_value = peak_value - np.min(values)  # Range
        l1_normal = np.sum(np.abs(values))  # L1 Normal
        l2_normal = np.sqrt(np.sum(np.square(values)))  # L2 Normal

        # Store results in the dictionary
        features_dict[column] = {
            'Mean': mean_value,
            'Standard Deviation': std_value,
            'Kurtosis': kurtosis_value,
            'Peak Value': peak_value,
            'Peak-to-Peak': peak_to_peak_value,
            'Impulse Factor': impulse_factor,
            'skewness_value': skewness_value,
            'crest_factor': crest_factor,
            'shape_factor':shape_factor,
            'rms_value': rms_value,
            'range_value': range_value,
            'l1_normal': l1_normal,
            'l2_normal': l2_normal
        }
    return features_dict
def create_correlation_heatmap(features_data, selected_features):
    """Create a correlation heatmap from extracted features."""
    # Calculate correlation matrix
    correlation_matrix = features_data[selected_features].corr()
    
    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        text=np.around(correlation_matrix.values, decimals=2),  # Show values on hover
        texttemplate='%{text}',  # Format for hover text
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        width=800,
        height=800,
        xaxis={'tickangle': 45}  # Rotate x-axis labels for better readability
    )
    
    return fig

# Global variable to store top correlated features
TOP_CORRELATED_FEATURES = []

# finding top correlated features 
def find_top_correlated_features(correlation_matrix, n=5):
    global TOP_CORRELATED_FEATURES
    
    # Get upper triangle of correlation matrix
    corr_matrix = correlation_matrix.abs().where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Unstack and sort
    corr_pairs = corr_matrix.unstack()
    sorted_corr = corr_pairs.sort_values(kind="quicksort", ascending=False)
    
    # Filter out perfect correlations (1.0)
    sorted_corr = sorted_corr[sorted_corr < 1.0]
    
    # Get top n correlations
    top_correlations = sorted_corr.head(n)
    
    # Store in global variable
    TOP_CORRELATED_FEATURES = list(top_correlations.index)
    
    # Display results
    st.subheader("Top 5 Correlated Features")
    for pair, correlation in top_correlations.items():
        st.write(f"{pair[0]} - {pair[1]}: {correlation:.4f}")
    
    return TOP_CORRELATED_FEATURES
    
# Feature selection block
st.sidebar.subheader("Feature Extraction Options")
select_all_features = st.sidebar.checkbox("Select All Features", value=False)
selected_features = st.sidebar.multiselect(
    "Select Features to Extract",
    options=list(feature_options.keys()),
    disabled=select_all_features
)

if select_all_features:
    selected_features = list(feature_options.keys())
    
window_size_input = st.sidebar.text_input("Enter window size for feature extraction and coorelation", value="1024")

# Extract Features button functionality
if st.button("Extract Features"):
    if not selected_files:
        st.warning("Please select at least one file for feature extraction.")
    elif not selected_features:
        st.warning("Please select at least one feature to extract.")
    else:
        all_features_data = []  # Store features from all files
        
        for selected_file in selected_files:
            data_path = os.path.join(output_folder, selected_file)
            try:
                # Load the selected CSV file
                data_frame = pd.read_csv(data_path)

                # Extract features from the DataFrame with a specified window size (e.g., 1000)
                window_size = int(window_size_input)  # Adjust as needed
                features_data = extract_features_from_dataframe(data_frame, window_size)
                features_data_combined = extract_features_for_all_data_points(data_frame)
                
                # Display extracted features
                st.subheader(f"Features extracted from {selected_file}")
                features_df = pd.DataFrame.from_dict(features_data_combined, orient='index')
                st.dataframe(features_df.style.background_gradient(cmap='YlGnBu'))
                
                # Store features for correlation analysis
                all_features_data.append(features_data)

            except FileNotFoundError:
                st.error(f"File not found: {data_path}. Please ensure the file exists in the dataset_csv folder.")
                continue
        
        # Combine all feature data into a single DataFrame for correlation analysis
        if all_features_data:
            combined_features_data = pd.concat(all_features_data, ignore_index=True)

            # Create and display correlation heatmap if we have data
            if len(combined_features_data) > 1:  # Only create heatmap if we have multiple columns
                st.subheader("Feature Correlation Analysis")
                
                # Create and display the correlation heatmap
                correlation_fig = create_correlation_heatmap(combined_features_data, selected_features)
                st.plotly_chart(correlation_fig)
                
                # Display correlation matrix as a table
                st.subheader("Correlation Matrix")
                correlation_matrix = combined_features_data[selected_features].corr()
                top_features = find_top_correlated_features(correlation_matrix)
                st.dataframe(correlation_matrix.round(2))
                                            
# Fast Fourier Transform section
st.sidebar.subheader("Fast Fourier Transform Options")
select_fft_all_csv = st.sidebar.checkbox("Select All CSV Files for FFT", value=False)
fft_selected_csv_files = st.sidebar.multiselect("Select CSV Files to Perform FFT", options=csv_files, disabled=select_fft_all_csv)

if select_fft_all_csv:
    fft_selected_csv_files = csv_files  # Automatically select all files if "Select All" is checked
if st.button("Calculate FFT"):
    if not fft_selected_csv_files:
        st.warning("Please select at least one CSV file to perform FFT.")
    else:
        for csv_file in fft_selected_csv_files:
            # Load the selected CSV file
            data_path = os.path.join(output_folder, csv_file)
            df_fft = pd.read_csv(data_path)

            # Assuming we want to perform FFT on the first column of each DataFrame
            values_fft = df_fft[df_fft.columns[0]].values

            # Calculate FFT
            fft_values = fft(values_fft)

            # Frequency axis calculation with scaling for 6000 Hz max
            N = len(values_fft)  # Number of samples
            max_frequency = 6000 # Target maximum frequency for the x-axis
            freq_axis = np.linspace(0, max_frequency, N // 2)  # Scale frequency axis to 6000 Hz

            try:
                # Plotting FFT results using Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=freq_axis, y=np.abs(fft_values[:N // 2]), mode='lines', name=csv_file))
                fig.update_layout(
                    title=f'FFT of {csv_file}',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Magnitude',
                    hovermode='x unified',
                    xaxis=dict(
                        rangeslider=dict(visible=True),  # Add range slider for zoom
                        rangeselector=dict(  # Add range selector buttons
                            buttons=list([
                                dict(count=1, label="1s", step="second", stepmode="backward"),
                                dict(count=10, label="10s", step="second", stepmode="backward"),
                                dict(count=100, label="100s", step="second", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    ), 
                    yaxis=dict(fixedrange=False)  # Allow zoom on y-axis
                )

                # Show the combined plot within Streamlit using Plotly's interface.
                st.plotly_chart(fig)  # Display the plot directly within Streamlit's app

            except ZeroDivisionError:
                st.error("Error calculating sample spacing. Ensure that plot duration and number of samples are valid.")   

                                 
# Plotting section.
st.sidebar.subheader("Plotting Options")
select_plot_all_csv = st.sidebar.checkbox("Select All CSV Files for Plotting", value=False)
plot_selected_csv_files = st.sidebar.multiselect("Select CSV Files to Plot", options=csv_files, disabled=select_plot_all_csv)

if select_plot_all_csv:
    plot_selected_csv_files = csv_files  # Automatically select all files if "Select All" is checked

if st.button("Plot Selected Data"):
    if not plot_selected_csv_files:
        st.warning("Please select at least one CSV file to plot.")
    else:
        # Prepare DataFrames and column names for plotting
        dataframes_to_plot = []
        column_names_to_plot = []

        for csv_file in plot_selected_csv_files:
            data_path = os.path.join(output_folder, csv_file)
            df_to_plot = pd.read_csv(data_path)

            dataframes_to_plot.append(df_to_plot)
            column_names_to_plot.append(df_to_plot.columns[0])  # Assuming we want to plot the first column

        # Initialize a figure for plotting with user-defined duration and sampling rate
        fig = go.Figure()

        # Loop through each DataFrame to create a combined plot using user-defined parameters
        for i, df in enumerate(dataframes_to_plot):
            values = df[column_names_to_plot[i]].values

            # Generate time axis based on user-defined total duration and sampling rate
            time_points = np.linspace(0, float(plot_duration_input), int(float(plot_duration_input) * int(sampling_rate_input)))

            # Ensure data length matches time points (trim or pad as necessary)
            if len(values) < len(time_points):
                time_points = time_points[:len(values)]  # Trim time_points if necessary
            else:
                values = values[:len(time_points)]  # Trim the data if it's longer than required

            # Normalize color intensity based on max value with less aggressive scaling
            max_value = np.max(values)

            # Calculate color intensity but ensure it doesn't go too low
            max_overall_value = np.max([np.max(df[column_names_to_plot[j]].values) for j in range(len(dataframes_to_plot))])
            
            color_intensity = max(0.3, 1 - (max_value / max_overall_value) ** 0.5)  # Set a minimum alpha of 0.3
            
            # Define color based on intensity and selected color (you can customize these colors)
            colors = ['#FF5733', '#33FF57', '#3357FF', '#8E44AD']  # Vivid Red, Green, Blue, Purple
            
            base_color = colors[i % len(colors)]
            
            rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)  # Use the calculated intensity
            rgba_color_str = f'rgba({rgba_color[0]*255},{rgba_color[1]*255},{rgba_color[2]*255},{rgba_color[3]})'
            # Plot data vs. time with a specific color and label using simple lines
            fig.add_trace(go.Scatter(x=time_points, y=values,
                                     mode='lines',
                                     name=f"{plot_selected_csv_files[i]}",
                                     line=dict(color=rgba_color_str, width=2)))

        # Set Y-axis limits with some padding based on all datasets
        y_min = min(np.min(df[column_names_to_plot[i]]) for i, df in enumerate(dataframes_to_plot))
        y_max = max(np.max(df[column_names_to_plot[i]]) for i, df in enumerate(dataframes_to_plot))

        fig.update_yaxes(range=[y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1])

        # Finalize plot settings
        fig.update_layout(
            title="Combined Time Plot of Selected Datasets",
            xaxis_title="Time (seconds)",
            yaxis_title="Data Values",
            hovermode='x unified',
            plot_bgcolor = 'white',
            xaxis=dict(
                rangeslider=dict(visible=True),  # Add range slider for zoom
                rangeselector=dict(  # Add range selector buttons
                    buttons=list([
                        dict(count=1, label="1s", step="second", stepmode="backward"),
                        dict(count=10, label="10s", step="second", stepmode="backward"),
                        dict(count=30, label="30s", step="second", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            ),
            yaxis=dict(fixedrange=False)  # Allow zoom on y-axis
        )

        # Show the combined plot within Streamlit using Plotly's interface.
        st.plotly_chart(fig)  # Display the plot directly within Streamlit's app

# Wavelet Transform options in the sidebar
st.sidebar.subheader("Wavelet Transform Options")
select_wavelet_all_csv = st.sidebar.checkbox("Select All CSV Files for Wavelet Transform", value=False)
wavelet_selected_csv_files = st.sidebar.multiselect("Select CSV Files for Wavelet Transform", options=csv_files, disabled=select_wavelet_all_csv)

# Dropdown menu for selecting the wavelet type
wavelet_types = ['Haar', 'Daubechies', 'Coiflets', 'Symlets', 'Morlet']
selected_wavelet = st.sidebar.selectbox("Select Wavelet Type", wavelet_types)

# Map user-friendly names to PyWavelets names
wavelet_map = {
    'Haar': 'haar',
    'Daubechies': 'db4',  # Use 'db4' for Daubechies
    'Coiflets': 'coif1',  # Use 'coif1' for Coiflets
    'Symlets': 'sym5',    # Use 'sym5' for Symlets
    'Morlet': 'morl'
}

if select_wavelet_all_csv:
    wavelet_selected_csv_files = csv_files  # Automatically select all files if "Select All" is checked

if st.button("Calculate Wavelet Transform"):
    if not wavelet_selected_csv_files:
        st.warning("Please select at least one CSV file to perform Wavelet Transform.")
    else:
        for csv_file in wavelet_selected_csv_files:
            # Load the selected CSV file
            data_path = os.path.join(output_folder, csv_file)
            try:
                df_wavelet = pd.read_csv(data_path)

                # Assuming we perform Wavelet Transform on the first column
                values_wavelet = df_wavelet[df_wavelet.columns[0]].values

                # Define Wavelet parameters
                wavelet_name = wavelet_map[selected_wavelet]

                if selected_wavelet == 'Morlet':
                    try:
                        # Perform Continuous Wavelet Transform (CWT)
                        scales = np.arange(1, 128)  # Scale range for the transform
                        coefficients, frequencies = pywt.cwt(values_wavelet, scales, wavelet_name)

                        # Generate a time axis for the data
                        time_points = np.linspace(0, len(values_wavelet) / int(sampling_rate_input), len(values_wavelet))

                        # Plot the Wavelet Transform as a heatmap using Plotly
                        fig = go.Figure(data=go.Heatmap(
                            z=np.abs(coefficients),
                            x=time_points,
                            y=frequencies,
                            colorscale='Viridis'
                        ))

                        fig.update_layout(
                            title=f"Wavelet Transform of {csv_file}",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Frequency (Hz)",
                            hovermode='x unified',
                            plot_bgcolor='white',
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=1, label="1s", step="second", stepmode="backward"),
                                        dict(count=10, label="10s", step="second", stepmode="backward"),
                                        dict(count=30, label="30s", step="second", stepmode="backward"),
                                        dict(step="all", label="All")
                                    ])
                                )
                            ),
                            yaxis=dict(fixedrange=False)
                        )

                        # Show the plot within Streamlit
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error processing file {csv_file} with Morlet wavelet: {e}")
                else:
                    try:
                        # Perform Discrete Wavelet Transform (DWT)
                        coeffs = pywt.wavedec(values_wavelet, wavelet_name, level=None)
                        # Reconstruct the signal from the wavelet coefficients
                        reconstructed_signal = pywt.waverec(coeffs, wavelet_name)

                        # Generate a time axis for the data
                        time_points = np.linspace(0, len(values_wavelet) / int(sampling_rate_input), len(reconstructed_signal))

                        # Plot the original signal using Plotly
                        fig_original = go.Figure()
                        fig_original.add_trace(go.Scatter(x=time_points, y=values_wavelet,
                                                          mode='lines',
                                                          name='Original Signal',
                                                          line=dict(color='blue', width=2),
                                                          opacity=0.8))

                        fig_original.update_layout(
                            title=f"Original Signal of {csv_file}",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Amplitude",
                            hovermode='x unified',
                            plot_bgcolor='white',
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=1, label="1s", step="second", stepmode="backward"),
                                        dict(count=10, label="10s", step="second", stepmode="backward"),
                                        dict(count=30, label="30s", step="second", stepmode="backward"),
                                        dict(step="all", label="All")
                                    ])
                                )
                            ),
                            yaxis=dict(fixedrange=False)
                        )

                        # Show the original signal plot within Streamlit
                        st.plotly_chart(fig_original)

                        # Plot the reconstructed signal using Plotly
                        fig_reconstructed = go.Figure()
                        fig_reconstructed.add_trace(go.Scatter(x=time_points, y=reconstructed_signal,
                                                               mode='lines',
                                                               name='Reconstructed Signal',
                                                               line=dict(color='red', width=2, dash='dash'),
                                                               opacity=0.5))

                        fig_reconstructed.update_layout(
                            title=f"Reconstructed Signal of {csv_file}",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Amplitude",
                            hovermode='x unified',
                            plot_bgcolor='white',
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=1, label="1s", step="second", stepmode="backward"),
                                        dict(count=10, label="10s", step="second", stepmode="backward"),
                                        dict(count=30, label="30s", step="second", stepmode="backward"),
                                        dict(step="all", label="All")
                                    ])
                                )
                            ),
                            yaxis=dict(fixedrange=False)
                        )

                        # Show the reconstructed signal plot within Streamlit
                        st.plotly_chart(fig_reconstructed)

                        # Plot the original and reconstructed signals together using Plotly
                        fig_combined = go.Figure()
                        fig_combined.add_trace(go.Scatter(x=time_points, y=values_wavelet,
                                                          mode='lines',
                                                          name='Original Signal',
                                                          line=dict(color='blue', width=2),
                                                          opacity=0.8))
                        fig_combined.add_trace(go.Scatter(x=time_points, y=reconstructed_signal,
                                                          mode='lines',
                                                          name='Reconstructed Signal',
                                                          line=dict(color='red', width=2, dash='dash'),
                                                          opacity=0.5))

                        fig_combined.update_layout(
                            title=f"Original and Reconstructed Signal of {csv_file}",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Amplitude",
                            hovermode='x unified',
                            plot_bgcolor='white',
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=1, label="1s", step="second", stepmode="backward"),
                                        dict(count=10, label="10s", step="second", stepmode="backward"),
                                        dict(count=30, label="30s", step="second", stepmode="backward"),
                                        dict(step="all", label="All")
                                    ])
                                )
                            ),
                            yaxis=dict(fixedrange=False)
                        )

                        # Show the combined plot within Streamlit
                        st.plotly_chart(fig_combined)
                    except Exception as e:
                        st.error(f"Error processing file {csv_file} with {selected_wavelet} wavelet: {e}")

            except Exception as e:
                st.error(f"Error processing file {csv_file}: {e}")
                
def calculate_features(window_data):
    """Calculate statistical features for a given window of data"""
    # Basic statistical features
    mean_value = np.mean(window_data)
    std_value = np.std(window_data)
    rms_value = np.sqrt(np.mean(np.square(window_data)))
    
    # Peak-based features
    peak_value = np.max(np.abs(window_data))
    peak_to_peak_value = np.max(window_data) - np.min(window_data)
    range_value = peak_to_peak_value
    
    # Shape features
    kurtosis_value = kurtosis(window_data)
    skewness_value = skew(window_data)
    
    # Factor calculations
    crest_factor = peak_value / rms_value if rms_value != 0 else 0
    shape_factor = rms_value / np.mean(np.abs(window_data)) if np.mean(np.abs(window_data)) != 0 else 0
    impulse_factor = peak_value / np.mean(np.abs(window_data)) if np.mean(np.abs(window_data)) != 0 else 0
    
    # Norm calculations
    l1_normal = np.linalg.norm(window_data, ord=1)
    l2_normal = np.linalg.norm(window_data, ord=2)
    
    # Combine all features
    features = np.array([
        mean_value, std_value, kurtosis_value, peak_value,
        peak_to_peak_value, impulse_factor, skewness_value,
        crest_factor, shape_factor, rms_value, range_value,
        l1_normal, l2_normal
    ])
    
    return features

def create_roc_curve(y_true, y_pred_proba):
    """Create ROC curve using Plotly"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='royalblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        width=700,
        height=500
    )
    return fig

def create_confusion_matrix(y_true, y_pred):
    """Create confusion matrix using Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        width=600,
        height=500
    )
    return fig

def create_tsne_plot(features, labels):
    """Create t-SNE visualization using Plotly"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'Label': labels
    })
    
    fig = px.scatter(
        df, x='x', y='y',
        color='Label',
        title='t-SNE Visualization of Features',
        labels={'color': 'Class'},
        width=700,
        height=500
    )
    
    fig.update_traces(marker=dict(size=8))
    return fig

# Define activation-loss pairs
ACTIVATION_LOSS_PAIRS = {
    "softmax": {
        "loss": CategoricalCrossentropy(),
        "description": "Multi-class classification",
        "preprocessing": "one-hot"
    },
    "sigmoid": {
        "loss": BinaryCrossentropy(),
        "description": "Binary classification",
        "preprocessing": "binary"
    }
}

# Sidebar options for file selection
st.sidebar.subheader("File Selection for SCL for feature and SVM for classification")
csv_files = [f for f in os.listdir("dataset_csv") if f.endswith('.csv')]

select_all_csv = st.sidebar.checkbox(
    "Select All CSV Files for Processing", 
    value=False, 
    key="select_all_csv_checkbox"
)

selected_csv_files = st.sidebar.multiselect(
    "Select CSV Files for Training",
    options=csv_files,
    disabled=select_all_csv,
    key="csv_file_multiselect"
)

if select_all_csv:
    selected_csv_files = csv_files

# Model Configuration in Sidebar
activation_function = st.sidebar.selectbox(
    "Select Activation Function",
    options=list(ACTIVATION_LOSS_PAIRS.keys()),
    index=0
)

st.sidebar.info(ACTIVATION_LOSS_PAIRS[activation_function]["description"])

# Direct Input Fields
epochs = st.sidebar.text_input("Enter number of epochs", value="50")
batch_size = st.sidebar.text_input("Enter batch size", value="32")
window_size = st.sidebar.text_input("Enter window size (number of points per window)", value="1024")

if activation_function == "softmax":
    num_classes = st.sidebar.text_input("Enter number of classes", value="2")
else:
    num_classes = "2"  # Fixed for sigmoid

# Convert input values
try:
    epochs = int(epochs)
    batch_size = int(batch_size)
    window_size = int(window_size)
    num_classes = int(num_classes)
except ValueError:
    st.error("Please enter valid integer values.")

# Display the selected files
st.write(f"### Selected CSV Files: {', '.join(selected_csv_files) if selected_csv_files else 'None'}")

# Enhanced Display Section
st.write("## Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.info("### Data Parameters")
    st.markdown("""
    ##### Selected CSV Files:
    ```
    {}
    ```
    """.format('\n'.join(selected_csv_files) if selected_csv_files else 'None'))
    
    st.markdown(f"""
    ##### Window Size:
    ```
    {window_size} points
    ```
    """)

with col2:
    st.info("### Model Parameters")
    params_df = pd.DataFrame({
        'Parameter': ['Activation Function', 'Loss Function', 'Epochs', 'Batch Size'],
        'Value': [
            activation_function,
            ACTIVATION_LOSS_PAIRS[activation_function]["loss"].__class__.__name__,
            epochs,
            batch_size
        ]
    })
    st.table(params_df)

st.markdown("---")

# Data Processing Section
if selected_csv_files:
    data_points = []
    for file in selected_csv_files:
        data_path = os.path.join("dataset_csv", file)
        data = pd.read_csv(data_path, header=None)
        data_points.extend(data.values.flatten())
    
    # Modified window processing with feature extraction
    X = []
    y = []
    for i in range(len(data_points) - window_size):
        window = data_points[i:i + window_size]
        features = calculate_features(window)
        X.append(features)
        y.append(data_points[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    binarizer = Binarizer(threshold=np.median(y))
    y = binarizer.fit_transform(y.reshape(-1, 1)).flatten()
    y = y.astype(int)
    
    if ACTIVATION_LOSS_PAIRS[activation_function]["preprocessing"] == "one-hot":
        if len(np.unique(y)) != num_classes:
            st.error(f"Data contains {len(np.unique(y))} classes but {num_classes} were specified!")
            st.stop()
        y = np.eye(num_classes)[y]
    else:  # binary
        if len(np.unique(y)) != 2:
            st.error("Data must have exactly 2 classes for binary classification!")
            st.stop()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1:]

    with st.expander("Show Data Statistics", expanded=False):
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric(label="Total Data Points", value=f"{len(data_points):,}")
            st.metric(label="Training Samples", value=f"{len(X_train):,}")
            st.metric(label="Number of Classes", value=num_classes)
        
        with col4:
            st.metric(label="Window Size", value=window_size)
            st.metric(label="Testing Samples", value=f"{len(X_test):,}")
            st.metric(label="Input Shape", value=f"{input_shape}")

    def create_model(input_shape, num_classes, activation_function):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        if activation_function == "softmax":
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        else:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
        return models.Model(inputs, outputs)

    model = create_model(input_shape, num_classes, activation_function)
    model.compile(
        optimizer='adam',
        loss=ACTIVATION_LOSS_PAIRS[activation_function]["loss"],
        metrics=['accuracy']
    )

    if st.sidebar.button("Train Model"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_loss'
        )
        
        class_weights = None
        if activation_function == "sigmoid":
            unique, counts = np.unique(y_train, return_counts=True)
            class_weights = dict(zip(unique, len(y_train) / (len(unique) * counts)))
        elif activation_function == "softmax":
            class_weights = dict(enumerate([1.0] * num_classes))
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )
        
        feature_extractor = models.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
        X_train_features = feature_extractor.predict(X_train, verbose=0)
        X_test_features = feature_extractor.predict(X_test, verbose=0)

        svm_classifier = SVC(kernel='rbf', gamma='scale', probability=True)
        if activation_function == "softmax":
            svm_classifier.fit(X_train_features, y_train.argmax(axis=1))
        else:
            svm_classifier.fit(X_train_features, y_train)

        y_pred = svm_classifier.predict(X_test_features)
        if activation_function == "softmax":
            y_true = y_test.argmax(axis=1)
        else:
            y_true = y_test
            
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        st.success(f"### Model Training Completed!")
        st.write(f"### Accuracy: {accuracy:.4f}")
        st.write("### Classification Report:")
        st.table(pd.DataFrame(report).transpose())
        
        # Performance Metrics Visualization
        st.write("### Performance Metrics")
        
        metric_tabs = st.tabs(["ROC Curve", "Confusion Matrix", "t-SNE Visualization"])
        
        with metric_tabs[0]:
            y_pred_proba = svm_classifier.predict_proba(X_test_features)[:, 1]
            roc_fig = create_roc_curve(y_true, y_pred_proba)
            st.plotly_chart(roc_fig)
        
        with metric_tabs[1]:
            conf_matrix_fig = create_confusion_matrix(y_true, y_pred)
            st.plotly_chart(conf_matrix_fig)
        
        with metric_tabs[2]:
            tsne_fig = create_tsne_plot(X_test_features, y_true)
            st.plotly_chart(tsne_fig)
        
        st.write("### Training History")
        history_df = pd.DataFrame(history.history)
        
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(
            y=history_df['loss'],
            name='Training Loss',
            line=dict(color='blue')
        ))
        fig_history.add_trace(go.Scatter(
            y=history_df['val_loss'],
            name='Validation Loss',
            line=dict(color='red')
        ))
        fig_history.update_layout(
            title='Training and Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            width=700,
            height=400
        )
        st.plotly_chart(fig_history)