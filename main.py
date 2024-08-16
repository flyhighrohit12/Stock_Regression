import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Set page config
st.set_page_config(page_title="Stock Market Analysis App", layout="wide")

# Load data function
@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
        if data['Date'].isnull().any():
            st.error('Some dates could not be parsed. Please check the date format and ensure it matches dd-mm-yyyy.')
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Main app
def main():
    st.title("Stock Market Analysis App")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if not data.empty:
            tabs = st.tabs(["How to Use", "EDA", "Training & Evaluation", "Predict"])

            with tabs[0]:
                how_to_use()

            with tabs[1]:
                eda(data)

            with tabs[2]:
                training_evaluation(data)

            with tabs[3]:
                predict(data)

# How to Use
def how_to_use():
    st.header("How to Use This Application")
    st.markdown("""
    - **Explore Data (EDA)**: View and interact with your data. Visualize closing prices and more.
    - **Training & Evaluation**: Train the Linear Regression model and evaluate its performance.
    - **Predict**: Input features to predict stock closing prices. Ensure you have trained the model first.
    """, unsafe_allow_html=True)

# Exploratory Data Analysis
def eda(data):
    st.header("Exploratory Data Analysis")
    st.write("Sample Data:")
    st.dataframe(data.head())
    fig = px.line(data, x='Date', y='Close', title='Closing Prices Over Time')
    st.plotly_chart(fig, use_container_width=True)

# Training and Evaluation
def training_evaluation(data):
    st.header("Training & Evaluation")
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean Squared Error:", mse)
    st.write("R^2 Score:", r2)
    joblib.dump(model, 'linear_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Prediction tab
def predict(data):
    st.header("Predict")
    if os.path.exists('linear_model.pkl') and os.path.exists('scaler.pkl'):
        model = joblib.load('linear_model.pkl')
        scaler = joblib.load('scaler.pkl')
        open_price = st.number_input("Open Price", value=data['Open'].mean())
        high_price = st.number_input("High Price", value=data['High'].mean())
        low_price = st.number_input("Low Price", value=data['Low'].mean())
        volume = st.number_input("Volume", value=int(data['Volume'].mean()))
        if st.button("Predict"):
            input_data = np.array([[open_price, high_price, low_price, volume]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            st.write(f"Predicted Close Price: ${prediction[0]:.2f}")
    else:
        st.error("Model not found. Please train the model in the 'Training & Evaluation' tab first.")

if __name__ == "__main__":
    main()
