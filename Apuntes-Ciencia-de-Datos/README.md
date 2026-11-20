# Apuntes de Ciencia de Datos

Repositorio de apuntes, cuadernos y pequeñas aplicaciones del curso de competencias en Ciencia de Datos e Inteligencia Artificial. Cubre la progresión desde el perceptrón simple hasta las redes neuronales convolucionales, con material de repaso para las evaluaciones. El contenido se organiza por cortes académicos.

## Contenido

### Primer corte

| Carpeta | Descripción |
| :-- | :-- |
| [`Perceptron/`](Perceptron) | Fundamentos del perceptrón: pesos, sesgo, regla de actualización, funciones de activación y descenso del gradiente. Incluye cuadernos con Scikit-learn, Keras y TensorFlow. |
| [`Redes densas/`](Redes%20densas) | Redes neuronales densas (MLP), regresión y clasificación, configuración de redes, redes convolucionales (CNN) y transferencia de aprendizaje. |
| [`prendas/`](prendas) | Aplicación en Streamlit que clasifica prendas de vestir dibujadas por el usuario, con un modelo Keras entrenado. |
| [`taller/`](taller) | Taller de procesamiento de datos de pacientes y aplicación en Streamlit para predecir riesgo cardíaco con una red neuronal implementada a mano. |
| [`resumen_corte_uno/`](resumen_corte_uno) | Resúmenes en PDF del primer corte: glosario, fórmulas y teoría del parcial, además de insignias de AWS Academy. |
| [`imagenes/`](imagenes) | Imágenes usadas en la documentación (perceptrón, neurona, gradiente, funciones de activación, etc.). |

### Segundo corte

| Carpeta | Descripción |
| :-- | :-- |
| [`procesamiento de imagen conceptos finales/`](procesamiento%20de%20imagen%20conceptos%20finales) | Cuadernos de visión por computador: convolución, CNN, transferencia de aprendizaje, clasificación de imágenes con TensorFlow y Keras, detección de objetos con YOLOv8, segmentación de instancias con YOLO11, detección de puntos clave, EPP y matrículas de vehículos. |
| [`Autoencoder y GANS/`](Autoencoder%20y%20GANS) | Autoencoders (introducción, señales ECG y limpieza de imágenes), GAN con Keras, Stable Diffusion y una GAN aplicada a imágenes tipo cromo del Mundial 2026. |
| [`n8n/`](n8n) | Orquestación de agentes inteligentes con n8n: agente del clima, pipeline RAG con PostgreSQL y `pgvector`, chatbot con memoria y guardrail, y servidor auxiliar en Python. |
| [`taller-piano-y-numeros/`](taller-piano-y-numeros) | Entrega del taller de piano virtual controlado con las manos y del taller de reconocimiento de números (documento en Word y PDF con los enlaces a los repositorios y a la aplicación desplegada). |

## Temas cubiertos

- Perceptrón, SSE y MSE, épocas, SGD, batch y mini-batch.
- Descenso del gradiente y tasa de aprendizaje.
- Funciones de activación: escalón, sigmoide, lineal, tanh, ReLU y softmax.
- Funciones de pérdida y métricas para regresión y clasificación.
- Modelos secuenciales y API funcional en Keras.
- Sobreajuste y Early Stopping.
- Redes convolucionales y procesamiento digital de imágenes.
- Transferencia de aprendizaje.

Segundo corte:

- Detección de objetos y segmentación con YOLOv8 y YOLO11.
- Detección de puntos clave y reconocimiento de matrículas.
- Autoencoders para reducción de ruido y detección de anomalías en ECG.
- Redes generativas adversarias (GAN) y modelos de difusión.
- Agentes de IA, RAG y automatización de flujos con n8n.

## Cuadernos

### Primer corte

#### Perceptron

- `Avanzada_Cuaderno_1_ANN_El_Perceptron.ipynb`
- `Avanzada_Cuaderno_2_ANN_Red_Neuronal_sklearn_keras_tensorflow.ipynb`
- `Cuanderno_1_1_Perceptron_con_Sklearn_ipynb (1).ipynb`
- `codigo-para-crear-circulos-concentricos.ipynb` y el modelo `modelo_mlp_circulos_concentricos.pkl`
- `pruebagpu.ipynb`

#### Redes densas

- `Avanzada_Cuaderno_3` Regresión lineal con una red neuronal básica.
- `Avanzada_Cuaderno_4` Clasificación con redes densas.
- `Cuaderno_5_Modelo_Regresion_Gasolina`
- `Avanzada Cuaderno 6` CNN y procesamiento digital de imágenes.
- `Avanzada_Cuaderno_7` Redes neuronales convolucionales.
- `Avanzada Cuaderno 8` CNN y transferencia de aprendizaje.
- `Configuracion_de_una_red_neuronal.ipynb`

Cada carpeta de teoría cuenta con su propio `Readme.md` con explicaciones más detalladas.

### Segundo corte

#### Procesamiento de imagen

- `Avanzada_Cuaderno_7` y `Cuaderno_8` Redes convolucionales, números y transferencia de aprendizaje.
- `Avanzada_Cuaderno_9_Convolucion` y clasificación de imágenes con TensorFlow y Keras.
- `Cuaderno_9` Detección de objetos con YOLOv8 y transfer learning.
- `Cuaderno_9A` Matrículas de vehículos.
- `Cuaderno_9B` Detección de EPP con YOLOv8.
- `Cuaderno_9C` Segmentación de instancias con YOLO11.
- `Cuaderno_9D` Detección de puntos clave.

#### Autoencoder y GANS

- `Cuaderno_14` Introducción a los autoencoders.
- `Cuaderno_15A`, `15B` y `15C` Procesamiento de ECG y autoencoders sobre señales (incluye la base LTAFDB).
- `Cuaderno_15D` Limpieza de imágenes con dos autoencoders.
- `Cuaderno_19` GAN con Keras.
- `Avanzada_Taller_Stable_Diffusion.ipynb`
- `CCD_UNAB_GAN_Panini_FIFA_2026.ipynb`

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

### Agentes con n8n (`n8n/`)

La carpeta `n8n/clima-Biing26-UCC-Ibague/` contiene los flujos exportados de n8n (`Agente del Clima.json`, `Embedding documento RAG (1).json` y `CCD UNAB — Chatbot RAG con Memoria y Guardrail.json`), los scripts SQL para crear la base de datos y la tabla vectorial con `pgvector`, documentos de conocimiento para RAG, `ccd_server.py`, un cliente de MLflow y `app_streamlit_cloud.py`. Tiene su propio `README.md` con el detalle de cada práctica.

### Taller de piano virtual y números (`taller-piano-y-numeros/`)

Contiene la entrega en `.docx` y `.pdf` con los enlaces a los repositorios del piano virtual con manos y del taller de números, y a la aplicación desplegada en Streamlit Cloud.

## Requisitos

- Python 3.12 o superior
- TensorFlow y Keras
- NumPy, pandas, scikit-learn, matplotlib y joblib
- Streamlit (para las aplicaciones)
- Ultralytics (YOLO), para los cuadernos de detección y segmentación
- n8n y PostgreSQL con `pgvector` (para la carpeta `n8n`)

Los cuadernos pueden ejecutarse en Visual Studio Code, Jupyter o Google Colab.

## Uso

```bash
git clone https://github.com/adiacla/Apuntes-Ciencia-de-Datos.git
cd Apuntes-Ciencia-de-Datos
```

Abra los cuadernos `.ipynb` y ejecute las celdas en orden.
