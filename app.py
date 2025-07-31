import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    data = pd.read_csv("sonar_data.csv", header=None)
    return data

@st.cache_resource
def train_model(data):
    X = data.drop(60, axis=1)
    y = data[60]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# App layout
st.title("Rock vs Mine Prediction App")
st.write("Enter the 60 sonar features below to predict whether the object is a rock or a mine.")

data = load_data()
model = train_model(data)

# Create 60 input fields for the user
user_input = []
for i in range(60):
    val = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
    user_input.append(val)

# Predict
if st.button("Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    if prediction == 'R':
        st.success("The object is **a Rock** 🪨")
    else:
        st.success("The object is **a Mine** 💣")
