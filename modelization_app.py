import streamlit as st
import pandas as pd
import pickle

st.title("Data Modelization Dashboard")

st.sidebar.title("Choose a model")
option = st.sidebar.selectbox(
    "Choose a model:",
    [
        "Random forest regressor",
        "XGBoost",
        "Inverse problem",
        
    ],
)

file_path = 'data/data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.drop(columns=['ID', 'FR-O', 'FR-W', 'TLP'])
data.OUTPUT = data.OUTPUT.apply(lambda x: 1 if x == "YES" else 0)
data.BUFFER = data.BUFFER.astype(str).str.strip()
data.BUFFER = data.BUFFER.replace({'PBS\xa0': 'PBS'})
data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
data.SIZE = data.SIZE.astype(float)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype(str)
allowed_strings = ["PBS", "MQ", "YES", "NO", "Micromixer", "Droplet junction", "HSPC", "ESM"]
for col in data.columns:
    if data[col].dtype == 'object':
        for val in data[col].unique():
            if val not in allowed_strings:
                print(f"Invalid string found in column '{col}': {val}")


# 1. Random forest regressor
if option == "Random forest regressor":
    st.header("Random forest regressor")
    st.write("Using multiple decision trees in parallel and bagging this model provide robust predictions for `SIZE` and `PDI`.")
    MODEL_PATH = "models/random_forest_model.pkl"  
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    st.write(f"Loaded {MODEL_PATH}")
    ml = st.selectbox("ML", ["HSPC", "ESM"])
    chip = st.selectbox("CHIP", ["Micromixer", "Droplet junction"])
    tlp = st.number_input("TLP", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
    esm = st.number_input("ESM", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
    hspc = st.number_input("HSPC", value=3.75, min_value=0.0, max_value=100.0, step=0.1)
    chol = st.number_input("CHOL", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
    peg = st.number_input("PEG", value=1.25, min_value=0.0, max_value=100.0, step=0.1)
    tfr = st.number_input("TFR", value=1.0, min_value=0.0, max_value=100.0, step=0.1)
    frr = st.number_input("FRR", value=3.0, min_value=0.0, max_value=100.0, step=0.1)
    buffer = st.selectbox("BUFFER", ["PBS ", "MQ"])
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "ML": [ml],
            "CHIP": [chip],
            "ESM": [esm],
            "HSPC": [hspc],
            "CHOL": [chol],
            "PEG": [peg],
            "TFR ": [tfr],
            "FRR": [frr],
            "BUFFER": [buffer],
        })
        predictions = model.predict(input_data)
        size, pdi = predictions[0]
        st.subheader("Model predictions:")
        st.write(f"`SIZE`: {size:.2f}")
        st.write(f"`PDI`: {pdi:.2f}")
    
# 2. XGBoost
elif option == "XGBoost":
    st.header("XGBoost")
    st.write("eXtreme Gradient Boosting predictions for `SIZE` and `PDI`.")
    MODEL_PATH = "models/xgboost_model.pkl"  
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    st.write(f"Loaded {MODEL_PATH}")
    ml = st.selectbox("ML", ["HSPC", "ESM"])
    chip = st.selectbox("CHIP", ["Micromixer", "Droplet junction"])
    tlp = st.number_input("TLP", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
    esm = st.number_input("ESM", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
    hspc = st.number_input("HSPC", value=3.75, min_value=0.0, max_value=100.0, step=0.1)
    chol = st.number_input("CHOL", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
    peg = st.number_input("PEG", value=1.25, min_value=0.0, max_value=100.0, step=0.1)
    tfr = st.number_input("TFR", value=1.0, min_value=0.0, max_value=100.0, step=0.1)
    frr = st.number_input("FRR", value=3.0, min_value=0.0, max_value=100.0, step=0.1)
    buffer = st.selectbox("BUFFER", ["PBS ", "MQ"])
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
        st.subheader("Model predictions:")
        st.write(f"`SIZE`: {size:.2f}")
        st.write(f"`PDI`: {pdi:.2f}")

# 3. Inverse model
elif option == "Inverse problem":
    st.header("Inverse problem")
    st.write("Inverse problem solver: given target `SIZE` and `PDI` returns predictions for the other numerical features.")
    MODEL_PATH = "models/inverse_xgboost_model.pkl"  
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    st.write(f"Loaded {MODEL_PATH}")
    size = st.number_input("SIZE", value=118.0, min_value=0.0, max_value=500.0, step=0.1)
    pdi = st.number_input("PDI", value=0.33, min_value=0.0, max_value=1.0, step=0.01)
    if st.button("Predict"):
        input_data = pd.DataFrame({"SIZE": [size],"PDI": [pdi]})
        predictions = model.predict(input_data)
        predictions = np.abs(predictions)
        predictions = np.where(predictions < 0.5, 0, predictions)
        st.subheader("Predicted Numerical Features")
        st.write("Predicted values for ESM, HSPC, CHOL, PEG, TFR, and FRR:", predictions)
