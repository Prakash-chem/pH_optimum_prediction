import streamlit as st
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from huggingface_hub import hf_hub_download
import subprocess
import os
from io import StringIO
from PIL import Image

# Display the image
image = Image.open('pH.png')
st.image(image, use_column_width=True)

# Model Selection
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
        
        if instability_index <= 30:
            instability_gauge = "Stable"
        elif 30 < instability_index <= 40:
            instability_gauge = "Moderately Stable"
        else:
            instability_gauge = "Unstable"
        
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

# Upload CSV file
st.write("Upload your CSV file containing protein sequences:")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read the uploaded file directly into a DataFrame
    df = pd.read_csv(uploaded_file)

    st.write("Current DataFrame columns:", df.columns.tolist())
    
    if 'ID' not in df.columns or 'Sequence' not in df.columns:
        st.error("The CSV file must contain 'ID' and 'Sequence' columns.")
    else:
        results = df['Sequence'].apply(calculate_protein_properties)

        new_columns = ['Molecular Weight', 'Oxidized Extinction Coefficient', 'Isoelectric Point',
                       'Lysine Count', 'Arginine Count', 'Cysteine Count', 'Instability Index',
                       'Instability Gauge', 'Hydrophobicity']

        for i, col in enumerate(new_columns):
            df[col] = [result[i] for result in results]

        st.write("DataFrame after adding new columns:", df.head())
        
        input_file = 'input_sequences.csv'
        df[['ID', 'Sequence']].to_csv(input_file, index=False)
        output_file = 'optimalpH_results.csv'

        # Download model weights
        model_id = "Loganz97/optimalpH"
        model_files = ["model_kmers", "model_knn", "model_xgboost"]
        local_weights_paths = {}

        for model_file in model_files:
            local_path = hf_hub_download(repo_id=model_id, filename=f"weights/{model_file}")
            local_weights_paths[model_file] = local_path
        
        command = f"python3 code/predict.py --input_csv {input_file} --id_col ID --seq_col Sequence --model_fname {local_weights_paths[selected_model]} --output_csv {output_file}"
        
        st.write(f"Running command: {command}")
        
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            st.write("OptimalpH model output:", result.stdout)
            if result.stderr:
                st.error(f"Error output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred while running the OptimalpH model. Return code: {e.returncode}")
            st.error(f"Error output: {e.stderr}")

        if os.path.exists(output_file):
            optimalpH_results = pd.read_csv(output_file)
            optimalpH_results = optimalpH_results.rename(columns={'Predicted_pH': 'predict_optimal_pH'})
            df = pd.merge(df, optimalpH_results[['ID', 'predict_optimal_pH']], on='ID', how='left')
        else:
            st.error(f"Warning: OptimalpH output file '{output_file}' not found.")
        
        final_columns = ['ID', 'Sequence', 'Molecular Weight', 'Oxidized Extinction Coefficient',
                         'Isoelectric Point', 'predict_optimal_pH', 'Lysine Count', 'Arginine Count',
                         'Cysteine Count', 'Instability Index', 'Instability Gauge', 'Hydrophobicity']

        missing_cols = [col for col in final_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in the DataFrame: {missing_cols}")
        else:
            df = df[final_columns]

        st.write("Final DataFrame with results:", df)
        st.dataframe(df)
