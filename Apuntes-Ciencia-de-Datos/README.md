# Apuntes de Ciencia de Datos

Repositorio de apuntes, cuadernos y pequeñas aplicaciones del curso de competencias en Ciencia de Datos e Inteligencia Artificial. Cubre la progresión desde el perceptrón simple hasta las redes neuronales convolucionales, con material de repaso para las evaluaciones.

## Contenido

| Carpeta | Descripción |
| :-- | :-- |
| [`Perceptron/`](Perceptron) | Fundamentos del perceptrón: pesos, sesgo, regla de actualización, funciones de activación y descenso del gradiente. Incluye cuadernos con Scikit-learn, Keras y TensorFlow. |
| [`Redes densas/`](Redes%20densas) | Redes neuronales densas (MLP), regresión y clasificación, configuración de redes, redes convolucionales (CNN) y transferencia de aprendizaje. |
| [`prendas/`](prendas) | Aplicación en Streamlit que clasifica prendas de vestir dibujadas por el usuario, con un modelo Keras entrenado. |
| [`taller/`](taller) | Taller de procesamiento de datos de pacientes y aplicación en Streamlit para predecir riesgo cardíaco con una red neuronal implementada a mano. |
| [`resumen_corte_uno/`](resumen_corte_uno) | Resúmenes en PDF del primer corte: glosario, fórmulas y teoría del parcial, además de insignias de AWS Academy. |
| [`imagenes/`](imagenes) | Imágenes usadas en la documentación (perceptrón, neurona, gradiente, funciones de activación, etc.). |

## Temas cubiertos

- Perceptrón, SSE y MSE, épocas, SGD, batch y mini-batch.
- Descenso del gradiente y tasa de aprendizaje.
- Funciones de activación: escalón, sigmoide, lineal, tanh, ReLU y softmax.
- Funciones de pérdida y métricas para regresión y clasificación.
- Modelos secuenciales y API funcional en Keras.
- Sobreajuste y Early Stopping.
- Redes convolucionales y procesamiento digital de imágenes.
- Transferencia de aprendizaje.

## Cuadernos

### Perceptron

- `Avanzada_Cuaderno_1_ANN_El_Perceptron.ipynb`
- `Avanzada_Cuaderno_2_ANN_Red_Neuronal_sklearn_keras_tensorflow.ipynb`
- `Cuanderno_1_1_Perceptron_con_Sklearn_ipynb (1).ipynb`
- `codigo-para-crear-circulos-concentricos.ipynb` y el modelo `modelo_mlp_circulos_concentricos.pkl`
- `pruebagpu.ipynb`

### Redes densas

- `Avanzada_Cuaderno_3` Regresión lineal con una red neuronal básica.
- `Avanzada_Cuaderno_4` Clasificación con redes densas.
- `Cuaderno_5_Modelo_Regresion_Gasolina`
- `Avanzada Cuaderno 6` CNN y procesamiento digital de imágenes.
- `Avanzada_Cuaderno_7` Redes neuronales convolucionales.
- `Avanzada Cuaderno 8` CNN y transferencia de aprendizaje.
- `Configuracion_de_una_red_neuronal.ipynb`

Cada carpeta de teoría cuenta con su propio `Readme.md` con explicaciones más detalladas.

## Aplicaciones

### Predictor de prendas (`prendas/`)

Aplicación web en Streamlit. El usuario dibuja una prenda en un lienzo, la imagen se convierte a escala de grises de 28x28 y el modelo `prendas.keras` la clasifica en una de 10 categorías (camiseta, pantalón, jersey, vestido, abrigo, sandalia, camisa, zapatos, bolso, botas).

```bash
cd prendas
pip install -r requirements.txt
streamlit run app.py
```

### Predicción de riesgo cardíaco (`taller/`)

- `procesamiento.py`: limpia `pacientes.csv` (nulos y rangos válidos de edad y colesterol), ajusta la etiqueta a -1/1, estandariza las variables con `StandardScaler` y genera `datos_procesados.json`, `modelo_estandarizacion.joblib` y `grafico_dispersion.png`.
- `app.py`: aplicación Streamlit con una red neuronal de 22 neuronas (activaciones ReLU y tanh final) escrita explícitamente, que predice el riesgo a partir de edad y colesterol.

```bash
cd taller
pip install -r requirements.txt
python procesamiento.py
streamlit run app.py
```

## Requisitos

- Python 3.12 o superior
- TensorFlow y Keras
- NumPy, pandas, scikit-learn, matplotlib y joblib
- Streamlit (para las aplicaciones)

Los cuadernos pueden ejecutarse en Visual Studio Code, Jupyter o Google Colab.

## Uso

```bash
git clone https://github.com/adiacla/Apuntes-Ciencia-de-Datos.git
cd Apuntes-Ciencia-de-Datos
```

Abra los cuadernos `.ipynb` y ejecute las celdas en orden.
