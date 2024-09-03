import streamlit as st
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from huggingface_hub import hf_hub_download
import subprocess
import os
import sys
from PIL import Image

# Clone the repository (if needed)
if not os.path.exists('optimalpH_colab'):
    subprocess.run(['git', 'clone', 'https://github.com/Loganz97/optimalpH_colab.git'])

sys.path.append('optimalpH_colab')

# Display the image
image = Image.open('pH.png')
st.image(image, use_column_width=True)

model_selection = st.selectbox("Choose a model", ["XGBoost (Most Accurate)", "KNN (Also Performs Well)", "K-mers"])

model_map = {
    "XGBoost (Most Accurate)": "model_xgboost",
    "KNN (Also Performs Well)": "model_knn",
    "K-mers": "model_kmers"
}

selected_model = model_map[model_selection]

def calculate_protein_properties(sequence):
    try:
        if not isinstance(sequence, str):
            raise ValueError(f"Invalid sequence type: {type(sequence)}. Expected string.")
        
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
                arginine_count, cysteine_count, instability_index, instability_gauge,
                hydrophobicity)
    except Exception as e:
        st.error(f"Error processing sequence: {e}")
        return tuple([None] * 9)

st.write("Upload your CSV file containing protein sequences:")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Current DataFrame columns:", df.columns.tolist())
    
    if 'ID' not in df.columns or 'Sequence' not in df.columns:
        st.error("The CSV file must contain 'ID' and 'Sequence' columns.")
    else:
        results = df['Sequence'].apply(calculate_protein_properties)

        new_columns = ['Molecular
