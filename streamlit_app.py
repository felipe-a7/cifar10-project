import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input



# Project paths and settings


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TRANSFER_IMAGE_SIZE = (224, 224)


# Streamlit page setup


st.set_page_config(
    page_title="CIFAR-10 Live Demo",
    layout="wide"
)

st.title("CIFAR-10 Image Classification Demo")
st.write(
    "Upload an image, choose a model, and see the predicted CIFAR-10 class "
    "and confidence scores for all classes."
)



# Load models


@st.cache_resource
def load_model(model_path):
    """
    Load a trained Keras model only once.
    """
    if not os.path.exists(model_path):
        return None

    return tf.keras.models.load_model(model_path, compile=False)


MODEL_OPTIONS = {
    "Baseline CNN": {
        "path": os.path.join(OUTPUT_DIR, "CNN.keras"),
        "type": "basic"
    },
    "Improved CNN v3": {
        "path": os.path.join(OUTPUT_DIR, "CNN_improved_v3_final_patched.keras"),
        "type": "basic"
    },
    "Transfer Learning ResNet50": {
        "path": os.path.join(OUTPUT_DIR, "resnet50_cifar10.keras"),
        "type": "transfer"
    }
}



# Image preprocessing


def preprocess_image_for_model(image, model_type):
    """
    Preprocess uploaded image depending on model type.

    Baseline and Improved CNN:
    - resize to 32x32
    - normalize to [0, 1]

    Transfer Learning ResNet50:
    - resize to 224x224
    - apply ResNet50 preprocessing
    """
    image = image.convert("RGB")

    if model_type == "transfer":
        img = image.resize(TRANSFER_IMAGE_SIZE)
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
    else:
        img = image.resize((32, 32))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_image_for_display(image):
    """
    Resize image to 32x32 for display in CIFAR style.
    """
    image = image.convert("RGB")
    return image.resize((32, 32))



# Prediction

def predict_image(model, image_array):
    """
    Predict CIFAR-10 class probabilities.
    """
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CIFAR10_CLASSES[predicted_index]
    confidence = float(probabilities[predicted_index])

    return probabilities, predicted_index, predicted_class, confidence



# Sidebar controls


st.sidebar.header("Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose model",
    list(MODEL_OPTIONS.keys())
)

selected_model_info = MODEL_OPTIONS[selected_model_name]
selected_model_path = selected_model_info["path"]
selected_model_type = selected_model_info["type"]

model = load_model(selected_model_path)

if model is None:
    st.error(f"Model file not found: `{selected_model_path}`")
    st.stop()

st.sidebar.success(f"Loaded: {selected_model_name}")



# Upload image


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to start the live demo.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Uploaded Image")
    st.image(image, caption="Original uploaded image", use_container_width=True)

with col2:
    st.subheader("CIFAR-style Preview")
    st.image(
        preprocess_image_for_display(image),
        caption="Image resized to 32x32",
        width=180
    )



# Run prediction


image_array = preprocess_image_for_model(image, selected_model_type)

probabilities, predicted_index, predicted_class, confidence = predict_image(
    model,
    image_array
)

st.markdown("---")

st.subheader("Prediction Result")

st.success(
    f"Predicted class: **{predicted_class}** "
    f"with **{confidence * 100:.2f}%** confidence"
)



# Confidence bar chart


st.subheader("Confidence Scores")

confidence_df = pd.DataFrame({
    "Class": CIFAR10_CLASSES,
    "Confidence": probabilities
})

confidence_df = confidence_df.sort_values("Confidence", ascending=False)

st.bar_chart(
    confidence_df.set_index("Class")
)



# Demo note


st.markdown("---")

st.subheader("Demo Note")

st.info(
    "This live demo focuses on model prediction and confidence scores. "
    "Grad-CAM visualisations were generated separately during the evaluation stage "
    "and saved in the outputs folder for the report."
)


# Explanation

st.markdown("---")

st.subheader("What this demo shows")

st.write(
    """
    This app demonstrates the final model pipeline in an interactive way:

    - Upload any image.
    - The selected model predicts the CIFAR-10 class.
    - The confidence score for each class is shown as a bar chart.
    - The sidebar allows switching between the baseline CNN, improved CNN v3, and transfer learning model.
    """
)