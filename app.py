import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load column names (you must have saved them in Jupyter Notebook)
with open("columns.pkl", "rb") as file:
    data_columns = pickle.load(file)

# Extract location names (assuming first 3 columns are sqft, bath, bhk)
locations = data_columns[3:]

# Streamlit UI
st.title("House Price Prediction")

sqft = st.number_input("Enter Square Feet Area", min_value=100, max_value=10000, step=10)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Enter Number of BHK", min_value=1, max_value=10, step=1)

location = st.selectbox("Select Location", locations)

if st.button("Predict Price"):
    # Create input array matching the model's feature structure
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    # Set the correct location index to 1 (one-hot encoding)
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # Predict price
    price = model.predict([x])[0]
    st.success(f"Predicted House Price: ₹ {price:.2f} Lakh")
