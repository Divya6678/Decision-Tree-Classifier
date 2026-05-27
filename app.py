import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("heart.csv")

model = pickle.load(
    open(
        "heart_disease_model.pkl",
        "rb"
    )
)

# ==================================================
# TITLE
# ==================================================

st.title("❤️ Heart Disease Prediction System")

st.markdown(
"""
This application predicts whether a person is likely
to have heart disease using a Decision Tree Classifier.
"""
)

# ==================================================
# SIDEBAR
# ==================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "❤️ Heart Disease Analysis",
        "⭐ Feature Importance",
        "🔍 Prediction"
    ]
)

# ==================================================
# HOME
# ==================================================

if menu == "🏠 Home":

    st.header("Project Overview")

    st.write(
    """
    ### Objective

    Predict whether a patient has heart disease
    based on medical attributes.

    ### Machine Learning Algorithm

    - Decision Tree Classifier

    ### Target Variable

    - 0 → No Heart Disease
    - 1 → Heart Disease

    ### Features

    - Age
    - Sex
    - Chest Pain Type
    - Blood Pressure
    - Cholesterol
    - Heart Rate
    - Exercise Induced Angina
    - ST Depression
    and more...
    """
    )

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Target",
        "Heart Disease"
    )

# ==================================================
# DATASET OVERVIEW
# ==================================================

elif menu == "📊 Dataset Overview":

    st.header("Dataset Overview")

    st.subheader("First 5 Records")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

# ==================================================
# HEART DISEASE ANALYSIS
# ==================================================

elif menu == "❤️ Heart Disease Analysis":

    st.header("Heart Disease Analysis")

    # --------------------------
    # Target Distribution
    # --------------------------

    st.subheader(
        "Heart Disease Distribution"
    )

    fig1, ax1 = plt.subplots()

    sns.countplot(
        x="target",
        data=df,
        ax=ax1
    )

    st.pyplot(fig1)

    # --------------------------
    # Age Distribution
    # --------------------------

    st.subheader(
        "Age Distribution"
    )

    fig2, ax2 = plt.subplots()

    sns.histplot(
        df["age"],
        kde=True,
        ax=ax2
    )

    st.pyplot(fig2)

    # --------------------------
    # Chest Pain Type
    # --------------------------

    st.subheader(
        "Chest Pain Type vs Disease"
    )

    fig3, ax3 = plt.subplots()

    sns.countplot(
        x="cp",
        hue="target",
        data=df,
        ax=ax3
    )

    st.pyplot(fig3)

    # --------------------------
    # Heart Rate
    # --------------------------

    st.subheader(
        "Maximum Heart Rate"
    )

    fig4, ax4 = plt.subplots()

    sns.boxplot(
        x="target",
        y="thalach",
        data=df,
        ax=ax4
    )

    st.pyplot(fig4)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

elif menu == "⭐ Feature Importance":

    st.header(
        "Feature Importance"
    )

    importance = pd.DataFrame({

        "Feature":
        df.drop(
            "target",
            axis=1
        ).columns,

        "Importance":
        model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance
    )

    st.subheader(
        "Importance Chart"
    )

    fig5, ax5 = plt.subplots(
        figsize=(10,6)
    )

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        ax=ax5
    )

    st.pyplot(fig5)



# ==================================================
# PREDICTION
# ==================================================

else:

    st.header(
        "Heart Disease Prediction"
    )

    age = st.slider(
        "Age",
        20,
        80,
        40
    )

    sex = st.selectbox(
        "Sex",
        [0,1]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0,1,2,3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        80,
        250,
        120
    )

    chol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )

    restecg = st.selectbox(
        "Rest ECG",
        [0,1,2]
    )

    thalach = st.number_input(
        "Max Heart Rate",
        60,
        250,
        150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0,1]
    )

    oldpeak = st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

    slope = st.selectbox(
        "Slope",
        [0,1,2]
    )

    ca = st.selectbox(
        "CA",
        [0,1,2,3,4]
    )

    thal = st.selectbox(
        "Thal",
        [0,1,2,3]
    )

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    if st.button(
        "Predict"
    ):

        prediction = model.predict(
            input_data
        )

        if prediction[0] == 1:

            st.error(
                "⚠️ Heart Disease Detected"
            )

        else:

            st.success(
                "✅ No Heart Disease"
            )