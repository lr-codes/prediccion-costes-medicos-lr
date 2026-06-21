# Predicción de Costes Médicos de Seguros usando Machine Learning

Este proyecto implementa un modelo de **Regresión Lineal Múltiple** desde cero utilizando Python y `scikit-learn` para predecir el coste de la prima de un seguro médico en función de las características demográficas y de salud del paciente.

## Características del Proyecto
* **Análisis de Datos:** Uso de `pandas` para el preprocesamiento y limpieza de datos.
* **Feature Engineering (Ingeniería de Características):** * Creación de una variable de interacción (`riesgo_extremo`) para capturar el impacto combinado del Índice de Masa Corporal (BMI) y el tabaquismo, pues la mezcla de ambos dispara el riesgo de salud.
  * Implementación de una transformación polinómica manual (`age2`) para capturar la naturaleza no lineal del envejecimiento en los costes médicos, pues el coste médico no es el mismo pasando de 20 a 30 años que de 50 a 60 años. 
* **Cross Validation:** Se ha hecho uso de la validación cruzada con el fin de mejorar la fiabilidad del modelo. Se basa en separar los datos en 10 partes iguales en este caso, y entrenarlo con 4 de ellas y examinarlo con la otra. Se repite el proceso 10 veces y se toma la media de la precisión como medida final.

## Resultados del Modelo
* **Precisión (R² Score):** **~83.92%** de la variabilidad de los costes médicos explicada con éxito (una mejora notable frente al 75% del modelo base sin ingeniería de variables).
* **Precisión (Cross-Validation):** **~83.43%**. En este caso más fiable y robusta y menos influenciado por la "suerte" de la división.
* **Conclusión Clave:** El modelo determinó que el factor de mayor peso financiero es el tabaquismo. Esto se debe a la evolución cuadrática que posee la edad, frente a la lineal del tabaquismo. En el modelo original el mayor factor con un gran margen de diferencia era el tabaquismo. Sin embargo en la última versión no se puede asegurar por lo que se explica en la observación, ya que no se puede ver como afecta individualmente de manera directa.

## Observaciones
* Debido a la creación de variables cruzadas como ha sido `riesgo_extremo` ya los coeficientes no son fiables para ver cada uno individualmente. Esto oculta el balance que se hacen entre ellos pues las predicciones son válidas y correctas.

## Tecnologías Utilizadas
* Python 3
* Pandas
* Scikit-Learn