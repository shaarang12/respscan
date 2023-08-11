# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import resnet as rt
# from PIL import Image
# import pickle

# # Create an instance of the model architecture
# model = tf.keras.applications.ResNet50(weights=None, include_top=True, classes=1000)

# # Load the saved model weights using pickle
# with open('modelRESNET.pkl', 'rb') as f:
#     model_weights = pickle.load(f)
#     model.set_weights(model_weights)



# def preprocess_image(image):
#     image = image.resize((224, 224))  # Resize image to match model input size
#     image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image



# def main():
#     st.title("Image Classification App")
#     st.write("Upload an image for prediction")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)

#         if st.button("Predict"):
#             # Preprocess the image
#             input_image = preprocess_image(image)

#             processed_image = rt.preprocess_input(input_image.copy())

#             # Perform prediction
#             predictions = model.predict(processed_image)
#             predicted_class = np.argmax(predictions[0])

#             st.write(f"Predicted class: {class_labels[predicted_class]}")

# if __name__ == "__main__":
#     main()





import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet as rt
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Load the saved model
model = load_model('ModelWeights.h5')

# Define class labels
class_labels = ['ACA', 'N', 'SCC']  # Replace with your actual class labels

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input size
    imagen = img_to_array(image)  # Normalize pixel values to [0, 1]
    imageb = np.expand_dims(imagen, axis=0)  # Add batch dimension
    return imageb.copy()

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
            processed_image = rt.preprocess_input(input_image)
            # Perform prediction
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions, axis = 1)
            #Adenocarcinoma, Normal, Squamous Cell Carcinoma
            if(predicted_class == [0]):
                st.write(f"Predicted class: Adenocarcinoma")
            elif(predicted_class == [1]):
                st.write(f"Predicted class: Benign Tissue")
            else:
                st.write(f"Predicted class: Squamous Cell Carcinoma")

if __name__ == "__main__":
    main()
