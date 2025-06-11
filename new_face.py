import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

def main():
    st.title("🎭 Enhanced Face Detection App")
    st.markdown("""
    ## Welcome to the Face Detection App!
    This app uses the Viola-Jones algorithm to detect faces in your images.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Color picker for rectangle
        rect_color = st.color_picker(
            "Rectangle Color", 
            "#FF0000",  # Default red
            help="Choose the color for face detection rectangles"
        )
        
        # Convert hex color to BGR for OpenCV
        hex_color = rect_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        
        # Algorithm parameters
        st.subheader("Detection Parameters")
        scaleFactor = st.slider(
            "Scale Factor", 
            1.01, 2.0, 1.1, 0.01,
            help="Parameter specifying how much the image size is reduced at each image scale"
        )
        
        minNeighbors = st.slider(
            "Minimum Neighbors", 
            1, 10, 5,
            help="Parameter specifying how many neighbors each candidate rectangle should have"
        )
        
        minSize = st.slider(
            "Minimum Face Size (px)", 
            20, 200, 30,
            help="Minimum possible face size in pixels"
        )
    
    # Instructions section
    with st.expander("📖 How to use this app", expanded=True):
        st.markdown("""
        1. **Upload an image** using the file uploader below
        2. **Adjust detection parameters** in the sidebar:
           - Change rectangle color
           - Tune scale factor (1.01-2.0)
           - Adjust minimum neighbors (1-10)
           - Set minimum face size (20-200px)
        3. **View detected faces** in the processed image
        4. **Save the result** to your device if desired
        
        *Tip: For best results, use clear front-facing photos with good lighting.*
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Select an image file to analyze for faces"
    )
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        original_img = img_array.copy()
        
        # Convert to grayscale (Viola-Jones requires grayscale)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Load the pre-trained face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces with user parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=(minSize, minSize)
        )
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                img_array, 
                (x, y), 
                (x+w, y+h), 
                bgr_color, 
                2  # Thickness
            )
        
        # Display original and processed images
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(img_array, caption="Processed Image", use_container_width=True)
        
        # Display detection results
        st.success(f"✅ Detected {len(faces)} faces in the image!")
        
        # Save image feature
        if st.button("💾 Save Processed Image"):
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
            # Generate filename
            filename = f"output/processed_{uploaded_file.name}"
            
            # Convert back to BGR for OpenCV save
            save_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, save_img)
            
            st.success(f"Image saved as {filename}")
            st.balloons()

if __name__ == "__main__":
    main()