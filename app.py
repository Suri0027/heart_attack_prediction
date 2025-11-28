import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load model ---
model = pickle.load(open('model.pkl', 'rb'))

# --- App title ---
st.title("💓 Cardiovascular Disease Prediction Dashboard")

# Sidebar navigation
menu = st.sidebar.selectbox("Navegación", ["Inicio", "Dashboard", "Predicción"])

# --- Load dataset ---
df = pd.read_csv("cardio_train.csv", sep=';')
df['age_years'] = (df['age'] / 365).astype(int)

# =========================
# PANTALLA INICIO
# =========================

if menu == "Inicio":
    st.header("📌 Proyecto: Predicción de Enfermedad Cardiovascular")
    st.write("""
    Esta aplicación permite:
    - Visualizar estadísticas del dataset
    - Explorar factores de riesgo
    - Predecir si una persona tiene probabilidad de desarrollar una enfermedad cardiovascular  
    """)

# =========================
# DASHBOARD
# =========================

elif menu == "Dashboard":
    st.header("📊 Dashboard de Análisis Exploratorio")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Registros", df.shape[0])
    col2.metric("% Con Cardiopatía", f"{df['cardio'].mean()*100:.1f}%")
    col3.metric("Edad Promedio", f"{df['age_years'].mean():.1f} años")

    st.subheader("Distribución de Edad")
    fig1 = plt.figure(figsize=(7,4))
    sns.histplot(df['age_years'], kde=True, color='red')
    st.pyplot(fig1)

    st.subheader("Cardiopatía por Nivel de Colesterol")
    fig2 = plt.figure(figsize=(7,4))
    sns.countplot(x='cholesterol', hue='cardio', data=df)
    st.pyplot(fig2)

    st.subheader("Cardiopatía por Género")
    fig3 = plt.figure(figsize=(7,4))
    sns.countplot(x='gender', hue='cardio', data=df)
    st.pyplot(fig3)

# =========================
# PREDICCIÓN
# =========================

elif menu == "Predicción":
    st.header("🔮 Predicción del Riesgo Cardiovascular")
    st.write("Ingresa los datos del paciente:")

    age = st.slider("Edad", 20, 80, 45)
    gender = st.selectbox("Género", [1, 2])
    height = st.slider("Altura (cm)", 140, 200, 165)
    weight = st.slider("Peso (kg)", 45, 150, 70)
    ap_hi = st.slider("Presión Sistólica (ap_hi)", 80, 200, 120)
    ap_lo = st.slider("Presión Diastólica (ap_lo)", 50, 130, 80)
    cholesterol = st.selectbox("Colesterol", [1, 2, 3])
    gluc = st.selectbox("Glucosa", [1, 2, 3])
    smoke = st.selectbox("Fuma", [0, 1])
    alco = st.selectbox("Consume Alcohol", [0, 1])
    active = st.selectbox("Actividad Física", [0, 1])

    input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, 
                            cholesterol, gluc, smoke, alco, active]])

    if st.button("Predecir"):
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠ Alta probabilidad de enfermedad cardiovascular.")
        else:
            st.success("💚 Baja probabilidad de enfermedad cardiovascular.")
