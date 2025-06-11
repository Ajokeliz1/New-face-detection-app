import streamlit as st
import numpy as np
from PIL import Image
import os

# Fallback for OpenCV import
try:
    import cv2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.5.5.64"])
    import cv2

def main():
    st.title("🎭 Face Detection App")
    
    # Load Haar Cascade (with error handling)
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    except Exception as e:
        st.error(f"Failed to load face detector: {str(e)}")
        return
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        st.image(img_array, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()