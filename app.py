import numpy as np
import torch
from PIL import Image
import streamlit as st
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5\\runs\\train\\exp2\\weights\\last.pt', force_reload=True)

def detect_objects(image):
    results = model(image)

    detection_image = np.array(results.render()[0])
    return detection_image

def main():
    st.title("AI-Powered Awake-Drowsy Monitoring")
    
    stframe = st.empty()  # Placeholder for displaying video frame
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        pil_image = Image.fromarray(frame)
        detection_image = detect_objects(pil_image)
        
        # Convert BGR to RGB before displaying
        detection_image_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
        
        # Display the detection image in Streamlit
        stframe.image(detection_image_rgb, channels="RGB", use_column_width=True)
    
    cap.release()

if __name__ == "__main__":
    main()
