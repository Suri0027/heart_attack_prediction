import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------
# Cargar datos y entrenar el modelo (lo hacemos dentro de la app)
# -------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("cardio_train.csv", sep=";")
    df["age_years"] = (df["age"] / 365).astype(int)
    # limpiar valores imposibles de presión (por seguridad)
    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
    return df.reset_index(drop=True)

@st.cache_resource
def train_model(df):
    X = df[['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy

df = load_data()
model, accuracy = train_model(df)

# -------------------------------------------------------------------
# Configuración de la app
# -------------------------------------------------------------------

st.title("💓 Cardiovascular Disease Prediction Dashboard")

menu = st.sidebar.selectbox("Navegación", ["Inicio", "Dashboard", "Predicción"])

# -------------------------------------------------------------------
# INICIO
# -------------------------------------------------------------------
if menu == "Inicio":
    st.header("📌 Proyecto: Predicción de Enfermedad Cardiovascular")
    st.write("""
    Esta aplicación forma parte de un proyecto de Ciencia de Datos.

    **¿Qué hace la app?**
    - Muestra estadísticas del dataset de enfermedades cardiovasculares.
    - Permite explorar factores de riesgo (edad, colesterol, presión, etc.).
    - Predice si una persona tiene alta o baja probabilidad de presentar una enfermedad cardiovascular.

    **Modelo utilizado:** Random Forest Classifier  
    **Exactitud aproximada (accuracy):** {:.1f}%
    """.format(accuracy * 100))

# -------------------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------------------
elif menu == "Dashboard":
    st.header("📊 Dashboard de Análisis Exploratorio")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de registros", df.shape[0])
    col2.metric("% con cardiopatía", f"{df['cardio'].mean()*100:.1f}%")
    col3.metric("Edad promedio", f"{df['age_years'].mean():.1f} años")

    st.subheader("Distribución de la edad")
    fig1 = plt.figure(figsize=(7,4))
    sns.histplot(df['age_years'], kde=True)
    plt.xlabel("Edad (años)")
    st.pyplot(fig1)

    st.subheader("Cardiopatía por nivel de colesterol")
    fig2 = plt.figure(figsize=(7,4))
    sns.countplot(x='cholesterol', hue='cardio', data=df)
    plt.xlabel("Colesterol (1 = normal, 2 = alto, 3 = muy alto)")
    plt.legend(title="Cardiopatía", labels=["No", "Sí"])
    st.pyplot(fig2)

    st.subheader("Cardiopatía por género")
    fig3 = plt.figure(figsize=(7,4))
    sns.countplot(x='gender', hue='cardio', data=df)
    plt.xlabel("Género (1 = Mujer, 2 = Hombre)")
    plt.legend(title="Cardiopatía", labels=["No", "Sí"])
    st.pyplot(fig3)

# -------------------------------------------------------------------
# PREDICCIÓN
# -------------------------------------------------------------------
elif menu == "Predicción":
    st.header("🔮 Predicción del riesgo cardiovascular")
    st.write("Ingresa los datos de la persona para estimar el riesgo:")

    age = st.slider("Edad (años)", 20, 80, 50)
    gender = st.selectbox("Género", options=[1, 2], format_func=lambda x: "Mujer" if x == 1 else "Hombre")
    height = st.slider("Altura (cm)", 140, 210, 165)
    weight = st.slider("Peso (kg)", 40, 160, 70)
    ap_hi = st.slider("Presión sistólica (ap_hi)", 80, 200, 120)
    ap_lo = st.slider("Presión diastólica (ap_lo)", 50, 130, 80)
    cholesterol = st.selectbox("Colesterol", [1, 2, 3])
    gluc = st.selectbox("Glucosa", [1, 2, 3])
    smoke = st.selectbox("¿Fuma?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    alco = st.selectbox("¿Consume alcohol frecuentemente?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    active = st.selectbox("¿Realiza actividad física?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

    input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo,
                            cholesterol, gluc, smoke, alco, active]])

    if st.button("Predecir"):
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.write(f"Probabilidad estimada de cardiopatía: **{proba*100:.1f}%**")

        if pred == 1:
            st.error("⚠ Alta probabilidad de enfermedad cardiovascular.\nSe recomienda evaluación médica y cambios en el estilo de vida.")
        else:
            st.success("💚 Baja probabilidad de enfermedad cardiovascular.\nMantén hábitos saludables y revisiones periódicas.")
