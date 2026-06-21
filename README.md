# Predicción de Costes Médicos de Seguros usando Machine Learning

Este proyecto implementa un modelo de **Regresión Lineal Múltiple** desde cero utilizando Python y `scikit-learn` para predecir el coste de la prima de un seguro médico en función de las características demográficas y de salud del paciente.

## Características del Proyecto
* **Análisis de Datos:** Uso de `pandas` para el preprocesamiento y limpieza de datos.
* **Feature Engineering (Ingeniería de Características):** * Creación de una variable de interacción (`riesgo_extremo`) para capturar el impacto combinado del Índice de Masa Corporal (BMI) y el tabaquismo.
  * Implementación de una transformación polinómica manual (`age2`) para capturar la naturaleza no lineal del envejecimiento en los costes médicos, pues el coste médico no es el mismo pasando de 20 a 30 años que de 50 a 60 años. 

## 📊 Resultados del Modelo
* **Precisión (R² Score):** **~79.56%** de la variabilidad de los costes médicos explicada con éxito (una mejora notable frente al 75% del modelo base sin ingeniería de variables).
* **Conclusión Clave:** El modelo determinó que el factor de mayor peso financiero es la edad, seguido por el tabaquismo. Esto se debe a la evolución cuadrática que posee la edad, frente a la lineal del tabaquismo. En el modelo original el mayor factor con un gran margen de diferencia era el tabaquismo. 

## 🛠️ Tecnologías Utilizadas
* Python 3
* Pandas
* Scikit-Learn