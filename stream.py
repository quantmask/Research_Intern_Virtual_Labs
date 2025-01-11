import os
import scipy.io
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.fft import fft
import streamlit as st
import pywt
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import Binarizer

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
    'Mean': 'Mean',
    'Standard Deviation': 'Standard Deviation',
    'Kurtosis': 'Kurtosis',
    'Peak Value': 'Peak Value',
    'Peak-to-Peak': 'Peak-to-Peak',
    'Impulse Factor': 'Impulse Factor'
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

def extract_features_from_dataframe(dataframe):
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

        # Store results in the dictionary
        features_dict[column] = {
            'Mean': mean_value,
            'Standard Deviation': std_value,
            'Kurtosis': kurtosis_value,
            'Peak Value': peak_value,
            'Peak-to-Peak': peak_to_peak_value,
            'Impulse Factor': impulse_factor
        }

    return features_dict

# Feature selection block remains in the same region
st.sidebar.subheader("Feature Extraction Options")
select_all_features = st.sidebar.checkbox("Select All Features", value=False)
selected_features = st.sidebar.multiselect("Select Features to Extract", options=list(feature_options.keys()), disabled=select_all_features)
if select_all_features:
    selected_features = list(feature_options.keys())  # Automatically select all features if "Select All" is checked
# Extract Features button functionality

if st.button("Extract Features"):
    if not selected_files:
        st.warning("Please select at least one file for feature extraction.")
    elif not selected_features:
        st.warning("Please select at least one feature to extract.")
    else:
        for selected_file in selected_files:  # Iterate over the selected files
            # Construct the full path for the current file
            data_path = os.path.join(output_folder, selected_file)
            try:
                # Load the selected CSV file
                data_frame = pd.read_csv(data_path)

                # Extract features from the DataFrame
                features_data = extract_features_from_dataframe(data_frame)

                # Display extracted features
                st.subheader(f"Features extracted from {selected_file}")
                for column, metrics in features_data.items():
                    st.write(f"**Column: {column}**")
                    for metric in selected_features:
                        if metric in metrics:  # Check if the feature is selected by the user
                            st.write(f" {metric}: {metrics[metric]}")
                    st.write("-" * 40)

            except FileNotFoundError:
                st.error(f"File not found: {data_path}. Please ensure the file exists in the dataset_csv folder.")
 
                
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
                        
                         # Generate a time axis for the data
                        time_points = np.linspace(0, len(values_wavelet) / int(sampling_rate_input), len(values_wavelet))

                        # Combine coefficients into a single 2D array for heatmap
                        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

                        # Updated plot for the Wavelet Transform as a heatmap using Plotly
                        fig = go.Figure(data=go.Heatmap(
                            z=np.abs(coeff_arr),
                            x=time_points,
                            y=np.arange(coeff_arr.shape[0]),
                            colorscale='Viridis'
                        ))

                        fig.update_layout(
                            title=f"Wavelet Transform of {csv_file}",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Coefficient Index",
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
                
# Machine Learning section
# Function to load and preprocess the dataset
def load_and_preprocess_data(file_paths, window_size=1024):
    data_points = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None)
        data_points.extend(data.values.flatten())
    
    X = []
    y = []
    for i in range(len(data_points) - window_size):
        X.append(data_points[i:i + window_size])
        y.append(data_points[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    binarizer = Binarizer(threshold=np.median(y))
    y = binarizer.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y

# Function to create the neural network model
def create_model(input_shape, activation_function):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(256, activation=activation_function)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=activation_function)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation=activation_function)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation=activation_function)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    model = models.Model(inputs, outputs)
    return model

# Define the supervised contrastive loss
class SupervisedContrastiveLoss(Loss):
    def __init__(self, temperature=0.1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[0])
        logits = tf.linalg.matmul(y_pred, y_pred, transpose_b=True) / self.temperature
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))

# Streamlit application
st.title("Feature Extraction and Classification using Supervised Contrastive Learning & SVM")

# Sidebar options for file selection
st.sidebar.subheader("File Selection")
csv_files = [f for f in os.listdir("dataset_csv") if f.endswith('.csv')]
select_plot_all_csv = st.sidebar.checkbox("Select All CSV Files for Processing", value=False)
plot_selected_csv_files = st.sidebar.multiselect("Select CSV Files to Process", options=csv_files, disabled=select_plot_all_csv)

if select_plot_all_csv:
    plot_selected_csv_files = csv_files  # Automatically select all files if "Select All" is checked

# Sidebar options for activation function and epochs
st.sidebar.subheader("Model Parameters")
activation_function = st.sidebar.selectbox("Select Activation Function", ["relu", "elu", "softmax", "sigmoid", "prelu", "leaky_relu"])
epochs = st.sidebar.text_input("Enter number of epochs", value="10")

if st.button("Process Selected Data"):
    if not plot_selected_csv_files:
        st.warning("Please select at least one CSV file to process.")
    else:
        # Load and preprocess the selected files
        file_paths = [os.path.join("dataset_csv", file) for file in plot_selected_csv_files]
        X, y = load_and_preprocess_data(file_paths, window_size=1024)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Ensure the input shape is correct
        input_shape = X_train.shape[1:]
        st.write(f"Input shape: {input_shape}")
        
        # Create and compile the model
        model = create_model(input_shape, activation_function)
        model.compile(optimizer='adam', loss=SupervisedContrastiveLoss())
        
        # Use callbacks for early stopping, model checkpointing, and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        
        # Train the model with user-defined epochs and callbacks
        model.fit(X_train, y_train, epochs=int(epochs), batch_size=16, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])
        
        # Extract features from the trained model
        feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-2].output)
        X_train_features = feature_extractor.predict(X_train)
        X_test_features = feature_extractor.predict(X_test)
        
        # Train an SVM classifier on the extracted features
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train_features, y_train)
        
        # Predict and evaluate the SVM classifier
        y_pred = svm_classifier.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
        classification_report_df = pd.DataFrame(classification_report_dict).transpose()
        
        st.write(f"Input shape: {input_shape}")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write("Classification Report:")
        st.table(classification_report_df)