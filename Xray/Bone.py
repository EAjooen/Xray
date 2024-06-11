import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import os

def load_model_safely(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return load_model(model_path, compile=False)

def strip_numerical_labels(label):
    """Helper function to remove numerical labels and return the actual class name."""
    return ''.join([i for i in label if not i.isdigit()]).strip()

def object_detection_image():
    st.subheader("Please scroll down to see the processed image.")

    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        img1 = Image.open(file)
        img2 = np.array(img1)
        st.image(img1, caption="Uploaded Image")
        my_bar = st.progress(0)

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        try:
            # Load the initial model for left/right prediction
            model = load_model_safely("keras_model.h5")
            if model is None:
                return

            # Load the labels for left/right prediction
            labels_path = "labels.txt"
            if not os.path.exists(labels_path):
                st.error(f"Labels file not found: {labels_path}")
                return
            class_names = open(labels_path, "r").readlines()

            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Preprocess the image
            image = img1.convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array

            # Predict using the initial model for left/right
            prediction = model.predict(data)
            index = np.argmax(prediction)
            initial_class_name = strip_numerical_labels(class_names[index])
            confidence_score = prediction[0][index]

            # Load the secondary model for gender prediction
            gender_model = load_model_safely("keras_model_2.h5")
            if gender_model is None:
                return

            # Load the labels for gender prediction
            gender_labels_path = "labels_2.txt"
            if not os.path.exists(gender_labels_path):
                st.error(f"Labels file not found: {gender_labels_path}")
                return
            gender_class_names = open(gender_labels_path, "r").readlines()

            # Predict using the secondary model for gender
            gender_prediction = gender_model.predict(data)
            gender_index = np.argmax(gender_prediction)
            gender_class_name = strip_numerical_labels(gender_class_names[gender_index])
            gender_confidence_score = gender_prediction[0][gender_index]

            # Display results
            st.image(img2, caption=f"{initial_class_name} ({gender_class_name})")
            st.markdown(f"**Left/Right Prediction**: {initial_class_name} - {confidence_score:.2f}")
            st.markdown(f"**Gender Prediction**: {gender_class_name} - {gender_confidence_score:.2f}")
            my_bar.progress(100)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            my_bar.progress(0)

def main():
    st.markdown('<p style="font-size: 42px;">Welcome to Bones Detection App!</p>', unsafe_allow_html=True)
    
    choice = st.sidebar.selectbox("MODE", ("About", "Image"))
    
    if choice == "Image":
        object_detection_image()
    elif choice == "About":
        st.markdown("""
            This app uses a pre-trained neural network model to bones in images.
            Upload an image and see the prediction in action!
        """)

if __name__ == '__main__':
    main()
