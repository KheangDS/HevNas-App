import streamlit as st
from PIL import Image

from model.load_model import load_trained_model
from data.image_utils import load_image, get_test_transform
from inference.predict import predict_image

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "models/KlebJeb.pt"
CLASS_NAMES = ['Meat', 'Noodle-Pasta', 'Rice', 'Soup', 'Vegetable-Fruit']  # 🔁 change to your real classes
IMG_SIZE = 255

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="Image Classification App",
    layout="centered"
)

st.title("🧠 Image Classification App")
st.write("Upload an image and the model will predict its class.")

# -------------------------------
# LOAD MODEL (CACHE)
# -------------------------------
@st.cache_resource
def load_model():
    return load_trained_model(
        model_path=MODEL_PATH,
        num_classes=len(CLASS_NAMES)
    )

model = load_model()
transform = get_test_transform(IMG_SIZE)

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_tensor = load_image(uploaded_file, transform)

    # Predict
    label, confidence = predict_image(
        model=model,
        image_tensor=image_tensor,
        class_names=CLASS_NAMES
    )

    # Output
    st.markdown("### ✅ Prediction Result")
    st.success(f"**Class:** {label}")
    st.info(f"**Confidence:** {confidence:.2%}")
