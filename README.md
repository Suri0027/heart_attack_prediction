# 💓 Cardiovascular Disease Prediction Dashboard

Este proyecto forma parte de un proyecto final de Ciencia de Datos.  
El objetivo es predecir el riesgo de enfermedad cardiovascular a partir de datos clínicos básicos, usando un modelo de Machine Learning integrado en una aplicación de Streamlit.

## 🧬 Descripción del proyecto

- Dataset: *Cardiovascular Disease Dataset* (Kaggle).
- Registros: ~70,000 pacientes.
- Variables: edad, género, altura, peso, presión arterial, colesterol, glucosa, hábito de fumar, consumo de alcohol, actividad física, etc.
- Variable objetivo: `cardio` (1 = tiene enfermedad cardiovascular, 0 = no).

El flujo del proyecto es:

1. Carga y limpieza de datos.
2. Análisis exploratorio (EDA).
3. Entrenamiento de un modelo de clasificación (`RandomForestClassifier`).
4. Implementación del modelo en una app de Streamlit.
5. Despliegue en Streamlit Cloud.

## 🚀 App en Streamlit

La aplicación permite:

- Ver un **dashboard** con:
  - Total de registros.
  - Porcentaje de personas con enfermedad cardiovascular.
  - Distribución de la edad.
  - Gráficas por colesterol y género.

- Usar un **formulario de predicción** donde el usuario puede ingresar:
  - Edad, género, altura, peso.
  - Presión arterial sistólica y diastólica.
  - Niveles de colesterol y glucosa.
  - Si fuma, consume alcohol y realiza actividad física.

La app devuelve la **probabilidad estimada** de enfermedad cardiovascular y un mensaje interpretando el riesgo.

## 🛠️ Cómo correr el proyecto localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/Suri0027/heart_attack_prediction.git
cd heart_attack_prediction

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
