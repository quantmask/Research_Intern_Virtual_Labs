# Signal Processing and Machine Learning Pipeline

## Overview
This project implements an end-to-end pipeline for processing and analyzing signal data from Simulink models. It converts .mat files to CSV format, extracts various statistical features, performs signal processing transformations, and applies machine learning models for prediction using Supervised Contrastive Loss and Support Vector Machine (SVM).

## Features
- MAT to CSV conversion
- Feature extraction:
  - Statistical features (mean, median, kurtosis, peak-to-peak)
  - Wavelet transformation
  - Fast Fourier Transform (FFT)
  - Time-series visualization
- Machine Learning Models:
  - Supervised Contrastive Learning
  - Support Vector Machine (SVM)
- Interactive web interface using Streamlit

## Project Structure
```
project/
│
├── project.ipynb          # Main development notebook
├── stream.py             # Streamlit web application
├── requirements.txt      # Project dependencies
├── data_mat/                 # Directory for storing data files
│   ├── raw/             # Original .mat files
|── dataset_csv/       # Processed CSV files
└── models/              # Trained model files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/quantmask/Research_Intern_Virtual_Labs.git
cd Research_Intern_Virtual_Labs
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
To run the data processing and model training pipeline:
1. Open `project.ipynb` in Jupyter Notebook or JupyterLab
2. Follow the notebook cells sequentially
3. Adjust parameters as needed for your specific use case

### Streamlit Web Application
To run the web interface:
```bash
streamlit run stream.py
```

The application will be available at `http://localhost:8502`

## Data Processing Pipeline

1. **Data Loading**
   - Upload .mat files through the interface
   - Automatic conversion to CSV format
   - Initial data validation and cleaning

2. **Feature Extraction**
   - Statistical features computation
   - Time-domain analysis
   - Frequency-domain transformation (FFT)
   - Wavelet transformation

3. **Visualization**
   - Time-series plots
   - FFT spectrum analysis
   - Wavelet coefficients visualization
   - Feature correlation plots

4. **Machine Learning**
   - Data preprocessing and normalization
   - Supervised Contrastive Learning implementation
   - SVM model training and optimization
   - Performance evaluation and metrics

## Requirements
- Python 3.8+
- NumPy
- Pandas
- SciPy
- PyWavelets
- Scikit-learn
- TensorFlow
- Streamlit
- Matplotlib
- Seaborn

## Model Performance
The current implementation achieves the following performance metrics:
- Accuracy: [Add your accuracy]
- Precision: [Add your precision]
- Recall: [Add your recall]
- F1 Score: [Add your F1 score]

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments
- [Add any acknowledgments or references]
- [Credit any external libraries or resources used]

## Contact
Your Name - [quantmask@github.com]
Project Link: [https://github.com/quantmask/Research_Intern_Virtual_Labs]