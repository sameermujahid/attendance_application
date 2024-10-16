import os
import cv2
import torch
import pandas as pd
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import datetime

app = Flask(__name__)

# Load the YOLOv8 model (trained model)
model = YOLO('C:\\Users\\samee\\PycharmProjects\\webcamproject\\best.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Dictionary to keep track of recognized names and their timestamps
recognized_names = {}

# Initialize default deadline
deadline = "09:00"  # Default deadline time

# Brevo API Configuration
api_key = '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'
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = api_key

# Function to send email notification using Brevo
def send_brevo_notification(subject, content):
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
    sender = {"name": "Attendance System", "email": "sameermujahid7777@gmail.com"}
    to = [{"email": "sameermujahid7777@gmail.com"}]
    html_content = f"<html><body>{content}</body></html>"

    # Create email
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

# Function to save recognized names to a log file
def log_attendance(name, timestamp):
    formatted_time = timestamp.strftime("%H:%M:%S")
    formatted_date = timestamp.strftime("%d-%m-%Y")

    # Check if the name is already recorded
    if name not in recognized_names:
        recognized_names[name] = {"time": formatted_time, "date": formatted_date, "late": formatted_time > deadline}
    else:
        # Update the date and time if name already exists
        recognized_names[name]["time"] = formatted_time
        recognized_names[name]["date"] = formatted_date

# Generate video frames for real-time feed
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
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

        # Log attendance for new names
        for name in names_in_frame:
            if name not in recognized_names:
                log_attendance(name, datetime.datetime.now())

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to show video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to show attendance and the send button
@app.route('/')
def index():
    return render_template('index2.html', attendance=recognized_names, deadline=deadline)

# API to handle sending attendance email
@app.route('/send_attendance', methods=['POST'])
def send_attendance():
    if recognized_names:
        attendance_list = "<br>".join(f"{name} time: {info['time']} date: {info['date']}"
                                       for name, info in recognized_names.items())
        email_content = f"<h1>Attendance List</h1><ul>{attendance_list}</ul>"
        send_brevo_notification("Attendance List", email_content)
        return jsonify({"status": "success", "message": "Attendance list sent successfully!"}), 200
    return jsonify({"status": "error", "message": "No attendees to send."}), 400

# API to export attendance to Excel
@app.route('/export_attendance', methods=['POST'])
def export_attendance():
    if recognized_names:
        # Create a DataFrame from the recognized names dictionary
        df = pd.DataFrame.from_dict(recognized_names, orient='index')
        df.reset_index(inplace=True)
        df.columns = ['Name', 'Date', 'Time', 'Late']  # Rename columns

        # Generate filename based on the current date
        filename = f"attendance_{datetime.datetime.now().strftime('%Y-%m-%d')}.xlsx"

        # Save DataFrame to Excel
        df.to_excel(filename, index=False)

        return jsonify({"status": "success", "message": f"Attendance list saved as {filename}."}), 200

    return jsonify({"status": "error", "message": "No attendees to export."}), 400

# API to fetch the latest attendance list
@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    return jsonify(recognized_names)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
