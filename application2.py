import os
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import datetime
import base64

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

# Function to save recognized names to a log
def log_attendance(name, timestamp):
    formatted_time = timestamp.strftime("%H:%M:%S")
    formatted_date = timestamp.strftime("%d-%m-%Y")
    if name not in recognized_names:
        recognized_names[name] = {"time": formatted_time, "date": formatted_date, "late": formatted_time > deadline}

# Function to generate video frames for real-time feed
def generate_video_feed():
    video_html = """
    <video id="video" width="640" height="480" autoplay></video>
    <script>
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                document.getElementById('video').srcObject = stream;
            })
            .catch(function(error) {
                console.error('Unable to access the camera: ' + error);
            });
    </script>
    """
    st.components.v1.html(video_html, height=500)

# Streamlit application layout
st.title("Attendance System")
st.write("## Live Attendance")

# Start video feed
generate_video_feed()

# Static attendance list heading and buttons
st.subheader("Attendance List")
attendance_placeholder = st.empty()

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

# Main loop to process frames (simplified for illustration)
# In a real implementation, you would need to process frames for YOLO detection here

try:
    # Placeholder for a continuous update mechanism (not actual Streamlit behavior)
    while True:
        # This is where you would process frames using OpenCV and YOLO
        # For now, this will just serve the video feed
        pass
except Exception as e:
    st.error(f"An error occurred: {e}")
finally:
    # Cleanup actions if needed
    pass
