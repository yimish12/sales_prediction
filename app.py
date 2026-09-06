import streamlit as st
import joblib
import numpy as np

# 1. Load the pre-trained model
# Ensure 'linear_regression_model.sav' is in the same directory as this script
try:
    model = joblib.load('linear_regression_model.sav')
except FileNotFoundError:
    st.error("Model file not found! Please run your training script first to generate 'linear_regression_model.sav'.")
    st.stop()

# 2. Set up the Streamlit UI
st.title("Advertising Sales Prediction App 📈")
st.write("""
This app predicts the **Sales** based on advertising budgets for TV, Radio, and Newspaper.
Adjust the sliders below to see how different budgets affect predicted sales.
""")

st.sidebar.header("Input Advertising Budgets")

# 3. Create input widgets for the features (TV, Radio, Newspaper)
# Assuming typical ranges for the Advertising dataset (0 to 300)
tv_budget = st.sidebar.slider("TV Ad Budget ($)", min_value=0.0, max_value=350.0, value=5.0)
radio_budget = st.sidebar.slider("Radio Ad Budget ($)", min_value=0.0, max_value=100.0, value=4.0)
newspaper_budget = st.sidebar.slider("Newspaper Ad Budget ($)", min_value=0.0, max_value=150.0, value=2.0)

# Display the user inputs
st.subheader("Current Budgets Selected:")
st.write(f"- **TV:** ${tv_budget}")
st.write(f"- **Radio:** ${radio_budget}")
st.write(f"- **Newspaper:** ${newspaper_budget}")

# 4. Make Predictions
if st.button("Predict Sales"):
    # Format the input as a 2D array for the model
    input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    
    # Predict
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"### Predicted Sales: {prediction[0]:.2f} units")
