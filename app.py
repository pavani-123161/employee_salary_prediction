import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Streamlit App Title
st.title("Employee Salary Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Feature and Target selection
    all_columns = data.columns.tolist()

    X_columns = st.multiselect("Select Features (Independent variables):", all_columns)
    y_column = st.selectbox("Select Target (Dependent variable):", all_columns)

    if X_columns and y_column:
        X = data[X_columns]
    y = data[y_column]

    # Label Encoding for categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Model trained! Accuracy: {acc:.2f}")