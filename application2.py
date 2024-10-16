import os
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import datetime

# Load the YOLOv8 model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best.pt')  # Replace with your model path
model = YOLO(model_path)

# Dictionary to keep track of recognized names and their timestamps
recognized_names = {}
deadline = "09:00"  # Default deadline time

# Brevo API Configuration
api_key = 'your_api_key_here'
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = api_key

# Function to send email notification using Brevo
def send_brevo_notification(subject, content):
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
    sender = {"name": "Attendance System", "email": "your_email@example.com"}
    to = [{"email": "your_email@example.com"}]
    html_content = f"<html><body>{content}</body></html>"

    email = sib_api_v3_sdk.SendSmtpEmail(
        to=to,
        sender=sender,
        subject=subject,
        html_content=html_content
    )

    try:
        api_response = api_instance.send_transac_email(email)
        print(f"Email sent: {api_response}")
    except ApiException as e:
        print(f"Error sending email: {e}")

# Function to log attendance
def log_attendance(name, timestamp):
    formatted_time = timestamp.strftime("%H:%M:%S")
    formatted_date = timestamp.strftime("%d-%m-%Y")
    if name not in recognized_names:
        recognized_names[name] = {"time": formatted_time, "date": formatted_date, "late": formatted_time > deadline}

# Streamlit application layout
st.title("Attendance System")
st.write("## Live Attendance")

# Start video feed using OpenCV
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("No camera found or could not open the default camera.")
        st.stop()

    frame_placeholder = st.empty()

    # Process video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

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

        # Log attendance for new names (only once)
        for name in set(names_in_frame):
            log_attendance(name, datetime.datetime.now())

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update attendance data
        attendance_data = [{"Name": name, "Time": info["time"], "Date": info["date"], "Late": info["late"]}
                           for name, info in recognized_names.items()]
        attendance_df = pd.DataFrame(attendance_data)

        # Display attendance list
        st.subheader("Attendance List")
        st.dataframe(attendance_df)

        # Break the loop if the user stops the webcam
        if st.button("Stop Webcam"):
            cap.release()
            break

# Buttons for actions
col1, col2 = st.columns(2)
with col1:
    if st.button("Send Attendance Email", key="send_email"):
        if recognized_names:
            attendance_list_html = "<br>".join(f"{name} - Time: {info['time']} | Date: {info['date']}"
                                               for name, info in recognized_names.items())
            email_content = f"<h1>Attendance List</h1><ul>{attendance_list_html}</ul>"
            send_brevo_notification("Attendance List", email_content)
            st.success("Attendance list sent successfully!")
        else:
            st.error("No attendees to send.")

with col2:
    if st.button("Export Attendance", key="export_attendance"):
        if recognized_names:
            filename = f"attendance_{datetime.datetime.now().strftime('%Y-%m-%d')}.xlsx"
            attendance_df = pd.DataFrame.from_records(recognized_names.values())
            attendance_df.to_excel(filename, index=False)
            st.success(f"Attendance list saved as {filename}.")
        else:
            st.error("No attendees to export.")
