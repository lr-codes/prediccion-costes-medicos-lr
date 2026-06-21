import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

datos = pd.read_csv("insurance.csv")

print(datos.head())

# Pasar el yes y no de smoker a 0 y 1
datos['smoker'] = datos['smoker'].map({'yes':1, 'no':0})
datos['sex'] = datos['sex'].map({'male':1, 'female':0})

# Crear una nueva columna para riesgo extremo y curvar la edad, pues pasar de 20->30 no es igual de costoso que 50->60
datos['riesgo_extremo'] = datos['smoker'] * datos['bmi']
datos['age2'] = datos['age']**2

# Dobles corchetes para elegir qué es lo que vamos a intentar predecir
X = datos[['age', 'age2' , 'bmi', 'smoker', 'riesgo_extremo']]

# Columna objetivo a predecir
y = datos['charges']

print("--- VARIABLES PREDICTORAS (X) ---")
print(X.head())

print("--- VARIABLE BUSCADA (y) ---")
print(y.head())

# --- ENTRENAMIENTO DE MODELO ---
# Partir los datos para entrenar, no memorizar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

# Modelo es el objeto vacío
modelo = LinearRegression()
# Entrenamiento del modelo
modelo.fit(X_train, y_train)

print("--- LO QUE HA APRENDIDO EL MODELO ---")
print(f"Por cada año de Edad, el precio sube: {modelo.coef_[0]} dólares")
print(f"Por cada punto de BMI, el precio sube: {modelo.coef_[1]} dólares")
print(f"Si la persona es Fumadora, el precio sube: {modelo.coef_[2]} dólares")

# --- Evaluación del modelo y predicción con el 20% restante ---

# Primero medimos la precisión con el R^2 score
precision = modelo.score(X_test, y_test)
print("La precisión (R² Score) es de un " + str(precision*100) + "%")

# Hacemos una predicción con un paciente nuevo
p_nuevo = pd.DataFrame([[30, 22.3, 0]], columns=['age', 'bmi', 'smoker'])
p_nuevo['age2'] = p_nuevo['age']**2
p_nuevo['riesgo_extremo'] = p_nuevo['bmi'] * p_nuevo['smoker']

# Nos aeguramos de que sigue el mismo orden
p_nuevo = p_nuevo[['age', 'age2', 'bmi', 'smoker', 'riesgo_extremo']]

precio_estimado = modelo.predict(p_nuevo)
print("El precio estimado que predice el modelo para el paciente nuevo es de:" + str(precio_estimado) + "$")