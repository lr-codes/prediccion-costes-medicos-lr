import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar y preparar los datos (exactamente igual que en tu modelo)
datos = pd.read_csv("insurance.csv")
datos['smoker'] = datos['smoker'].map({'yes': 1, 'no': 0})
datos['sex'] = datos['sex'].map({'male': 1, 'female': 0})
datos['riesgo_extremo'] = datos['smoker'] * datos['bmi']
datos['age2'] = datos['age'] ** 2

# Nos quedamos solo con las columnas que nos interesan para el gráfico
datos_grafico = datos[['age', 'bmi', 'smoker', 'riesgo_extremo', 'charges']]

# --- GRÁFICO 1: MAPA DE CALOR ---
plt.figure(figsize=(10, 8))
# Calculamos la correlación y la pintamos
correlaciones = datos_grafico.corr()
sns.heatmap(correlaciones, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor de Correlaciones")
plt.show() # Esto pausa el código y abre una ventana con el gráfico

# --- GRÁFICO 2: MATRIZ DE DISPERSIÓN ---
print("Generando matriz de dispersión... (puede tardar unos segundos)")
# Usamos 'hue="smoker"' para que separe los colores por fumador/no fumador
sns.pairplot(datos_grafico, hue="smoker", palette="Set1", corner=True)
plt.suptitle("Relaciones entre variables (Rojo=Fumador, Gris=No Fumador)", y=1.02)
plt.show()