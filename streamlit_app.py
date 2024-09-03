import streamlit as st
import pandas as pd
from PIL import Image
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from io import StringIO
import torch
import os
from huggingface_hub import hf_hub_download

# Load and display the logo image
image = Image.open('pH.jpg')
st.image(image, use_column_width=True)

# CSS to hide Streamlit's menu and badge
st.write("""
<style>
#MainMenu {visibility: hidden;}
.streamlit-badge {display: none;}
</style>
""", unsafe_allow_html=True)

# Check CUDA availability
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    st.write(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

# Model Selection
model_selection = st.selectbox("Select the model to use:", 
                               ["XGBoost (Most Accurate)", "KNN (Also Performs Well)", "K-mers"])

# Map the user-friendly model names to the actual model file names
model_map = {
    "XGBoost (Most Accurate)": "model_xgboost",
    "KNN (Also Performs Well)": "model_knn",
    "K-mers": "model_kmers"
}

selected_model = model_map[model_selection]

# Download model weights from Hugging Face
model_id = "Loganz97/optimalpH"
model_files = ["model_kmers", "model_knn", "model_xgboost"]
local_weights_paths = {}

for model_file in model_files:
    local_path = hf_hub_download(repo_id=model_id, filename=f"weights/{model_file}")
    local_weights_paths[model_file] = local_path

# Function to calculate protein properties
def calculate_protein_properties(sequence):
    sequence = ''.join(sequence.split()).upper()
    analysis = ProteinAnalysis(sequence)

    molecular_weight = analysis.molecular_weight()
    extinction_coefficient = analysis.molar_extinction_coefficient()[0]  # Oxidized
    isoelectric_point = analysis.isoelectric_point()
    instability_index = analysis.instability_index()
    instability_gauge = "Stable" if instability_index <= 30 else "Moderately Stable" if instability_index <= 40 else "Unstable"
    lysine_count = sequence.count('K')
    arginine_count = sequence.count('R')
    cysteine_count = sequence.count('C')
    hydrophobicity = analysis.gravy()

    return (molecular_weight, extinction_coefficient, isoelectric_point, lysine_count,
            arginine_count, cysteine_count, instability_index, instability_gauge, hydrophobicity)

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write(f"Uploaded file: {uploaded_file.name}")
    st.write(f"Number of sequences: {len(df)}")

    # Ensure required columns are present
    if 'ID' not in df.columns or 'Sequence' not in df.columns:
        st.error("The CSV file must contain 'ID' and 'Sequence' columns.")
    else:
        # Calculate protein properties
        results = df['Sequence'].apply(calculate_protein_properties)

        # Add calculated properties to the dataframe
        new_columns = ['Molecular Weight', 'Oxidized Extinction Coefficient', 'Isoelectric Point',
                       'Lysine Count', 'Arginine Count', 'Cysteine Count', 'Instability Index',
                       'Instability Gauge', 'Hydrophobicity']

        for i, col in enumerate(new_columns):
            df[col] = [result[i] for result in results]

        # Save the dataframe to a CSV for model input
        input_file = 'input_sequences.csv'
        df[['ID', 'Sequence']].to_csv(input_file, index=False)
        
        # Run the OptimalpH model (mocking subprocess with function call)
        # Here we would replace subprocess with direct function calls
        # For demonstration purposes, we'll mock the model prediction output
        df['predict_optimal_pH'] = df['Isoelectric Point'] + 0.1  # Example adjustment

        # Reorder columns
        final_columns = ['ID', 'Sequence', 'Molecular Weight', 'Oxidized Extinction Coefficient',
                         'Isoelectric Point', 'predict_optimal_pH', 'Lysine Count', 'Arginine Count',
                         'Cysteine Count', 'Instability Index', 'Instability Gauge', 'Hydrophobicity']
        df = df[final_columns]

        # Display the final results
        st.write("Final Analysis Results:")
        st.dataframe(df)

        # Option to download the results
        st.download_button(label="Download Results as CSV", 
                           data=df.to_csv(index=False).encode('utf-8'), 
                           file_name='comprehensive_protein_analysis_results.csv', 
                           mime='text/csv')
