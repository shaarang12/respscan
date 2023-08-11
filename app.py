import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input

# Load the saved model
model = load_model('ModelWeights.h5')

# Define class labels
class_labels = ['ACA', 'N', 'SCC']  # Replace with your actual class labels

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input size
    image_array = np.array(image)
    processed_image = preprocess_input(image_array)  # Preprocess image using ResNet preprocessing
    return processed_image

def main():
    st.title("Image Classification App")
    st.write("Upload an image for prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict"):
            # Preprocess the image
            input_image = preprocess_image(image)
            input_batch = np.expand_dims(input_image, axis=0)

            # Perform prediction
            predictions = model.predict(input_batch)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_labels[predicted_class_idx]

            st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
