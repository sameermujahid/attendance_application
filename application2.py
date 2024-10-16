import os
import cv2
import pandas as pd
import streamlit as st
import datetime
import threading
from ultralytics import YOLO
import imageio
from PIL import Image

# Load the YOLOv8 model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best.pt')  # Replace with your model path
model = YOLO(model_path)

# Placeholder for displaying video frames
FRAME_WINDOW = st.image([])

# Dictionary to keep track of recognized names and their timestamps
recognized_names = {}
deadline = "09:00"  # Default deadline time

# Function to save recognized names to a log
def log_attendance(name, timestamp):
    formatted_time = timestamp.strftime("%H:%M:%S")
    formatted_date = timestamp.strftime("%d-%m-%Y")
    if name not in recognized_names:
        recognized_names[name] = {"time": formatted_time, "date": formatted_date, "late": formatted_time > deadline}

# Capture video frames using imageio
def capture_video():
    video = imageio.get_reader('<video0>')  # Access your default webcam

    for frame in video:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
        img = Image.fromarray(frame)

        # Process frame with YOLO model
        results = model.predict(frame)
        names_in_frame = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                name = model.names[cls]
                names_in_frame.append(name)

                label = f"{name}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Log attendance for new names
        for name in set(names_in_frame):
            log_attendance(name, datetime.datetime.now())

        # Display the frame in Streamlit
        FRAME_WINDOW.image(img)

# Start video capture in a separate thread
video_thread = threading.Thread(target=capture_video)
video_thread.start()

# Streamlit application layout
st.title("Attendance System")
st.write("## Live Attendance")

# Display the attendance list
st.subheader("Attendance List")
attendance_placeholder = st.empty()

# Buttons for actions
if st.button("Send Attendance Email"):
    # Code to send email (omitted here for brevity)
    st.success("Attendance list sent successfully!")

if st.button("Export Attendance"):
    # Code to export attendance (omitted here for brevity)
    st.success("Attendance list saved successfully.")
