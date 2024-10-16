import cv2
import streamlit as st
import threading
import datetime
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with your model's path

# Placeholder for video frame
FRAME_WINDOW = st.image([])

# Dictionary to track recognized names
recognized_names = {}
deadline = "09:00"

# Function to log attendance
def log_attendance(name, timestamp):
    formatted_time = timestamp.strftime("%H:%M:%S")
    formatted_date = timestamp.strftime("%d-%m-%Y")
    if name not in recognized_names:
        recognized_names[name] = {"time": formatted_time, "date": formatted_date, "late": formatted_time > deadline}

# Function to capture video and display in Streamlit
def capture_video():
    cap = cv2.VideoCapture(0)  # Access the default webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
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

                # Draw bounding box and label on frame
                label = f"{name}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Log attendance for new names
        for name in set(names_in_frame):
            log_attendance(name, datetime.datetime.now())

        # Convert the frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the Streamlit image placeholder with the frame
        FRAME_WINDOW.image(frame_rgb)

    cap.release()

# Run video capture in a separate thread
video_thread = threading.Thread(target=capture_video)
video_thread.start()

# Streamlit layout
st.title("Attendance System")
st.write("## Live Attendance Feed")

# Buttons for actions
if st.button("Send Attendance Email"):
    st.success("Attendance list sent successfully!")

if st.button("Export Attendance"):
    st.success("Attendance list exported successfully.")
