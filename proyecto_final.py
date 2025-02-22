import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Implementacion de la función sigmoidea
def sigmoidea(x):
    """
    Función de activación sigmoidea
    f(x) = 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-x))

#Esta es la clase PerceptronSigmoide
class PerceptronSigmoide:
    def __init__(self, n_inputs):
        """
        Inicializa el perceptrón
        Parámetros:
        n_inputs (int): Número de entradas (características) del perceptrón
        """
        # Se inicializan los pesos con valores aleatorios pequeños
        # Se agrega un peso adicional para el sesgo
        self.pesos = np.random.randn(n_inputs + 1) * 0.1

    def activar(self, entradas):
        """
        Devuelve la salida del perceptrón
        Parámetros:
        entradas (array): Vector de características de entrada
        Retorna:
        float: Salida del perceptrón después de aplicar la función sigmoidea
        """
        entradas = np.array(entradas)
        entradas_con_sesgo = np.insert(entradas, 0, 1)
        suma_ponderada = np.dot(entradas_con_sesgo, self.pesos)
        return sigmoidea(suma_ponderada)

    def entrenar(self, entradas, salidas_esperadas, tasa_aprendizaje=0.1, iteraciones=1000):
        """
        Parámetros:
        entradas (list): Lista de listas donde cada sublista es un conjunto de características
        salidas_esperadas (list): Lista de valores esperados de salida para cada conjunto de entradas
        tasa_aprendizaje (float): Tasa de aprendizaje para actualizar los pesos
        iteraciones (int): Número de iteraciones para el entrenamiento

        Retorna:
        list: Historia de errores durante el entrenamiento
        """
        X = np.array(entradas)
        y = np.array(salidas_esperadas)

        # Lista para almacenar la historia de errores
        error_history = []

        for _ in range(iteraciones):
            error_total = 0

            # Para cada ejemplo de entrenamiento
            for i in range(len(X)):
                # Calcular la salida actual
                salida_actual = self.activar(X[i])

                # Calcular el error
                error = y[i] - salida_actual
                error_total += error**2
                delta = error * salida_actual * (1 - salida_actual)

                # Actualizar el peso del sesgo
                self.pesos[0] += tasa_aprendizaje * delta

                # Actualizar los pesos de las entradas
                for j in range(len(X[i])):
                    self.pesos[j+1] += tasa_aprendizaje * delta * X[i][j]

            # Guardar el error cuadrático medio
            error_history.append(error_total / len(X))

        return error_history

# Se Generan los datos de entrenamiento
def generar_datos(n_muestras=100):
    """
    Se generan los datos sinteticos para una clasificacion binara para el plano bidimencional
    Parámetros:
    n_muestras (int): Número de muestras a generar

    Retorna:
    tuple: (X, y) donde X son las características y y las etiquetas
    """
    # Generamos puntos aleatorios en el plano
    X = np.random.rand(n_muestras, 2) * 2 - 1  # Valores entre -1 y 1

    # Definimos una línea para separar las clases: y = 0.5*x + 0.1
    # Los puntos por encima de la línea pertenecen a la clase 1, los de abajo a la clase 0
    y = np.array([1 if x2 > (0.5 * x1 + 0.1) else 0 for x1, x2 in X])

    return X, y

# Aqui gradicamos los resultados
def graficar_resultados(X, y, perceptron):
    """
    Aqui se grafican los puntos de entrenamiento profe XD
    Parámetros:
    X (array): Datos de características
    y (array): Etiquetas
    perceptron (PerceptronSigmoide): Perceptrón entrenado
    """
    # Crear una figura
    plt.figure(figsize=(10, 8))

    # Definir el rango de valores para x e y
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Crear una malla de puntos para evaluar el modelo
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Evaluar el perceptrón en cada punto de la malla
    Z = np.array([perceptron.activar([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Colorear la malla según las predicciones (con transparencia)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))

    # Dibujar la frontera de decisión (donde la salida es 0.5)
    plt.contour(xx, yy, Z, [0.5], linewidths=1, colors='k')

    # Graficar los puntos de entrenamiento
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', edgecolor='k', label='Clase 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='x', edgecolor='k', label='Clase 1')

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Perceptrón con función de activación sigmoidea')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Generar datos
    X, y = generar_datos(n_muestras=100)

    # Se Crea y entrenar el perceptrón
    perceptron = PerceptronSigmoide(n_inputs=2)
    historial_error = perceptron.entrenar(X, y, tasa_aprendizaje=0.1, iteraciones=1000)

    # Graficar los resultados
    graficar_resultados(X, y, perceptron)

    # Graficar la evolución del error durante el entrenamiento
    plt.figure(figsize=(10, 6))
    plt.plot(historial_error)
    plt.xlabel('Iteraciones')
    plt.ylabel('Error cuadrático medio')
    plt.title('Evolución del error durante el entrenamiento')
    plt.grid(True)
    plt.show()


    print(f"Pesos finales: Sesgo = {perceptron.pesos[0]:.4f}, w1 = {perceptron.pesos[1]:.4f}, w2 = {perceptron.pesos[2]:.4f}")
    print("\nPrueba con algunos ejemplos:")
    test_points = [[-0.5, -0.5], [0.5, 0.5], [0, 0]]
    for point in test_points:
        output = perceptron.activar(point)
        clase_predicha = 1 if output >= 0.5 else 0
        print(f"Punto {point}: Salida = {output:.4f}, Clase predicha = {clase_predicha}")

if __name__ == "__main__":
    main()
