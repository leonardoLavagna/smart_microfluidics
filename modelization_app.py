import streamlit as st
import pandas as pd
import pickle

st.title("Data Modelization Dashboard")

file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')
st.write("Dataset Preview:", data.head())

MODEL_PATH = "random_forest_model.pkl"  
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

st.write("Enter the features below to predict `SIZE` and `PDI`.")
ml = st.selectbox("ML", ["HSPC", "Other"])
chip = st.selectbox("CHIP", ["Micromixer", "Other"])
tlp = st.number_input("TLP", min_value=0.0, max_value=100.0, step=0.1)
esm = st.number_input("ESM", min_value=0.0, max_value=100.0, step=0.1)
hspc = st.number_input("HSPC", min_value=0.0, max_value=100.0, step=0.1)
chol = st.number_input("CHOL", min_value=0.0, max_value=100.0, step=0.1)
peg = st.number_input("PEG", min_value=0.0, max_value=100.0, step=0.1)
tfr = st.number_input("TFR", min_value=0.0, max_value=100.0, step=0.1)
frr = st.number_input("FRR", min_value=0.0, max_value=100.0, step=0.1)
fr_o = st.number_input("FR-O", min_value=0.0, max_value=100.0, step=0.1)
fr_w = st.number_input("FR-W", min_value=0.0, max_value=100.0, step=0.1)
buffer = st.selectbox("BUFFER", ["PBS ", "MQ"])
output = st.selectbox("OUTPUT", ["YES", "NO"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "ML": [ml],
        "CHIP": [chip],
        "TLP": [tlp],
        "ESM": [esm],
        "HSPC": [hspc],
        "CHOL": [chol],
        "PEG": [peg],
        "TFR ": [tfr],
        "FRR": [frr],
        "FR-O": [fr_o],
        "FR-W": [fr_w],
        "BUFFER": [buffer],
        "OUTPUT": [output]
    })

    predictions = model.predict(input_data)
    size, pdi = predictions[0]

    st.subheader("Predictions:")
    st.write(f"**SIZE**: {size:.2f}")
    st.write(f"**PDI**: {pdi:.2f}")
