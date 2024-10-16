import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import datetime

# Initialize the YOLO model
model = YOLO('yolov8n.pt')  # Ensure you have the model weights in the same directory

# Create an empty DataFrame to store attendance
attendance_list = pd.DataFrame(columns=['Name', 'Timestamp'])

# Function to recognize faces (dummy implementation)
def recognize_face(frame):
    # Placeholder for face recognition logic
    # For example, you can use a library like face_recognition
    return "John Doe"  # Replace this with your face recognition logic

# Start video capture
cap = cv2.VideoCapture(0)

st.title("Real-Time Video Feed with Attendance Tracking")

# Create a button to start attendance tracking
if st.button("Start Attendance Tracking"):
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Perform object detection with YOLO
        results = model(frame)

        # Process results for face detection
        for result in results:
            for det in result.boxes.xyxy:  # Extract bounding boxes
                x1, y1, x2, y2, conf, cls = det
                label = f'ID: {int(cls)} {conf:.2f}'  # Replace with your label or name
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Recognize face and mark attendance
                recognized_name = recognize_face(frame)  # Implement your face recognition logic
                if recognized_name not in attendance_list['Name'].values:
                    attendance_list.loc[len(attendance_list)] = [recognized_name, datetime.datetime.now()]

        # Display the video feed in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", use_column_width=True)

        # Display the attendance list
        st.subheader("Attendance List")
        st.dataframe(attendance_list)

        if st.button("Stop Attendance Tracking"):
            break

# Release the video capture when done
cap.release()
cv2.destroyAllWindows()
