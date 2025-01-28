import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('Fire_detection_R.h5')  # Ensure this path is correct
    return model

model = load_trained_model()

# Image preprocessing function
IMG_SIZE = (128, 128)  # Same size used during training

def preprocess_image(image):
    img = image.resize(IMG_SIZE)  # Resize image to match training input size
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize image pixels (between 0 and 1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit App UI
st.title("Forest Fire Detection")
st.subheader("Classify images into **Normal**, or **Fire_Detected** stages.")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Detect"):
        with st.spinner("Detecting..."):
            # Preprocess and predict
            image = load_img(uploaded_file)
            processed_image = preprocess_image(image)
            detection = model.predict(processed_image)
            
            # Class labels (ensure these match your model's output labels)
            class_labels = ['Fire detected', 'Normal']
            
            # Get detected class
            detected_class_index = np.argmax(detection, axis=1)[0]
            detected_label = class_labels[detected_class_index]
            
            # Confidence score (optional)
            confidence = np.max(detection) * 100  # Get highest probability for prediction
            
            # Adjust threshold if needed (example: only classify "Fire detected" if confidence > 75%)
            if detected_label == 'Fire detected' and confidence > 75:
                detected_label = "Uncertain - Normal Detected (low confidence)"
                
            else :detected_label = "Fire Detected (High confidence)"
            
            # Display detection result with confidence score
            st.success(f"Detected Class: **{detected_label}**")

