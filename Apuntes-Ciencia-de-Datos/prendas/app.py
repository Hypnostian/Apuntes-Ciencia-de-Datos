from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
	page_title="Predictor de prendas",
	page_icon="👕",
	layout="centered",
)


CLASS_NAMES = [
	"Camiseta/top",
	"Pantalón",
	"Jersey",
	"Vestido",
	"Abrigo",
	"Sandalia",
	"Camisa",
	"Zapatos",
	"Bolso",
	"Botas",
]
MODEL_PATH = Path(__file__).parent / "prendas.keras"
CANVAS_SIZE = 420


@st.cache_resource
def load_model():
	return tf.keras.models.load_model(MODEL_PATH, compile=False)


def prepare_image(image: Image.Image) -> np.ndarray:
	grayscale = ImageOps.grayscale(image)
	resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
	pixels = np.asarray(resized, dtype=np.float32) / 255.0
	return pixels


def predict(image: Image.Image):
	model = load_model()
	pixels = prepare_image(image)
	probabilities = model.predict(pixels[np.newaxis, ...], verbose=0)[0]
	predicted_index = int(np.argmax(probabilities))
	return pixels, predicted_index, probabilities


st.title("Predictor de prendas con TensorFlow")
st.write("Dibuja una prenda o carga una imagen para conocer la clase predicha.")

input_mode = st.radio("Fuente de la imagen", ["Dibujar", "Subir imagen"], horizontal=True)
image = None

if input_mode == "Dibujar":
	canvas_result = st_canvas(
		fill_color="rgba(255, 255, 255, 0)",
		stroke_width=14,
		stroke_color="#FFFFFF",
		background_color="#000000",
		height=CANVAS_SIZE,
		width=CANVAS_SIZE,
		drawing_mode="freedraw",
		key="prenda_canvas",
	)
	if canvas_result.image_data is not None:
		image = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("L")
else:
	uploaded_file = st.file_uploader(
		"Selecciona una imagen",
		type=["png", "jpg", "jpeg", "webp"],
	)
	if uploaded_file is not None:
		image = Image.open(uploaded_file).convert("L")
		st.image(image, caption="Imagen cargada", width=CANVAS_SIZE)

if st.button("Predecir prenda", type="primary", use_container_width=True):
	if image is None:
		st.warning("Dibuja una prenda o carga una imagen antes de predecir.")
	else:
		try:
			pixels, predicted_index, probabilities = predict(image)
			st.subheader(f"Predicción: {CLASS_NAMES[predicted_index]}")
			st.image(pixels, caption="Imagen procesada a 28 x 28 píxeles", width=180)

			probability_table = {
				"Prenda": CLASS_NAMES,
				"Probabilidad": [f"{value:.2%}" for value in probabilities],
			}
			st.dataframe(probability_table, hide_index=True, use_container_width=True)
		except Exception as error:
			st.error(f"No se pudo realizar la predicción: {error}")

st.divider()
st.subheader("Instrucciones")
st.markdown(
	"""
	- Dibuja con trazos blancos sobre el fondo negro o sube una imagen.
	- La imagen se convierte automáticamente a escala de grises y a 28 x 28 píxeles.
	- El modelo fue entrenado con imágenes similares a Fashion-MNIST: fondo negro y la prenda en tonos blancos.
	- Para obtener mejores resultados, utiliza imágenes similares o parecidas a las imágenes usadas durante el entrenamiento.
	- El modelo utiliza una salida `softmax` para asignar probabilidades a las diez clases.
	"""
)
