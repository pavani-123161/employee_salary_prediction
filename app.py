import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# App Title
st.title("🔍 Employee Salary Prediction using Machine Learning")

# Load dataset automatically (no upload)
df = pd.read_csv("data/employee_data.csv")
st.success("✅ Dataset loaded successfully from the repository.")

# Show data preview
st.write("### Data Preview")
st.write(df.head())

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
# Feature selection
st.write("### Select Features (Independent Variables):")
features = st.multiselect("Choose features (X):", options=df.columns)

# Target selection
st.write("### Select Target (Dependent Variable):")
target = st.selectbox("Choose target (y):", options=df.columns)

# Model training after selections
if features and target:
    X = df[features]
    y = df[target]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display results
    st.success(f"✅ Model Trained Automatically!")
    st.write(f"*Accuracy:* {accuracy * 100:.2f}%")