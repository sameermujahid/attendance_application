import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from ultralytics import YOLOimport os
import cv2
import json
import base64
import shutil
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from threading import Lock
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from waitress import serve
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
# ──────────────────────────────────────────────────────────────────────────────
# APP CONFIG
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Only ONE model — generic YOLOv8 face detector
# Used for: detecting faces in the attendance stream AND in enrollment frames
FACE_MODEL_PATH = "yolov8n-face-lindevs.pt"
face_model = YOLO(FACE_MODEL_PATH)

FACE_CONF        = 0.50   # face detection confidence
RECOG_THRESHOLD  = 0.72   # cosine similarity to accept a recognition match
DEADLINE         = "12:00"

attendance_lock = Lock()
enrollment_lock = Lock()

recognized_names: dict = {}
current_camera = None

# ── Enrollment storage ────────────────────────────────────────────────────────
ENROLLMENT_DIR = Path("enrolled_faces")
ENROLLMENT_DIR.mkdir(exist_ok=True)
ENROLLMENT_DB  = ENROLLMENT_DIR / "enrollment_db.json"

def load_enrollment_db() -> dict:
    if ENROLLMENT_DB.exists():
        with open(ENROLLMENT_DB, "r") as f:
            return json.load(f)
    return {}

def save_enrollment_db(db: dict):
    with open(ENROLLMENT_DB, "w") as f:
        json.dump(db, f, indent=2)

enrollment_db: dict = load_enrollment_db()

# ──────────────────────────────────────────────────────────────────────────────
# EMAIL (Brevo)
# ──────────────────────────────────────────────────────────────────────────────
api_key = os.getenv("BREVO_API_KEY", "PUT_YOUR_KEY_HERE")
_brevo_cfg = sib_api_v3_sdk.Configuration()
_brevo_cfg.api_key['api-key'] = api_key

def send_brevo_notification(subject: str, content: str):
    api = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(_brevo_cfg))
    email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": "mail"}],
        sender={"name": "Attendance System", "email": "mail"},
        subject=subject,
        html_content=content,
    )
    try:
        api.send_transac_email(email)
        print("Email sent successfully")
    except ApiException as e:
        print(f"Email error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# FACE DETECTION  (yolov8n-face-lindevs.pt)
# ──────────────────────────────────────────────────────────────────────────────

def detect_faces(image_bgr: np.ndarray) -> list:
    """
    Run yolov8n-face on image_bgr.
    Returns list of (x1, y1, x2, y2, conf) for every detected face.
    """
    results = face_model.predict(image_bgr, verbose=False, conf=FACE_CONF)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
    return boxes


def crop_face(image_bgr: np.ndarray, x1, y1, x2, y2, pad_ratio=0.12) -> np.ndarray:
    """Crop the face from image_bgr with a small padding margin."""
    h, w = image_bgr.shape[:2]
    pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    crop = image_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def detect_largest_face(image_bgr: np.ndarray):
    """
    Detect faces, return (crop, (x1,y1,x2,y2,conf)) for the largest one,
    or (None, None) if no face found.
    """
    boxes = detect_faces(image_bgr)
    if not boxes:
        return None, None
    # pick largest by area
    x1, y1, x2, y2, conf = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    crop = crop_face(image_bgr, x1, y1, x2, y2)
    return crop, (x1, y1, x2, y2, conf)

# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING  (multi-block histogram — fast, no extra model)
# ──────────────────────────────────────────────────────────────────────────────

def extract_embedding(face_bgr: np.ndarray):
    """
    Resize face to 128×128, equalise histogram, compute 16-block
    32-bin histogram descriptor and L2-normalise.
    Returns a Python float list (ready for JSON), or None on failure.
    """
    gray = cv2.cvtColor(cv2.resize(face_bgr, (128, 128)), cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    block = 32
    hists = []
    for ry in range(0, 128, block):
        for rx in range(0, 128, block):
            b = gray[ry:ry+block, rx:rx+block]
            h = cv2.calcHist([b], [0], None, [32], [0, 256]).flatten()
            hists.append(h)
    emb = np.concatenate(hists).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm < 1e-6:
        return None
    emb /= norm
    return emb.tolist()

# ──────────────────────────────────────────────────────────────────────────────
# FACE RECOGNITION  (compare embedding against enrolled DB)
# ──────────────────────────────────────────────────────────────────────────────

def recognize_face(frame_bgr: np.ndarray):
    """
    Detect the largest face in frame_bgr with yolov8n-face,
    extract its embedding, compare cosine-similarity against enrolled DB.
    Returns (first_name, similarity_score) or (None, 0.0).
    """
    global enrollment_db
    face_crop, bbox = detect_largest_face(frame_bgr)
    if face_crop is None:
        return None, 0.0

    q_emb = extract_embedding(face_crop)
    if q_emb is None:
        return None, 0.0

    q_vec = np.array(q_emb)
    best_name, best_sim = None, -1.0

    for roll, info in enrollment_db.items():
        embeddings = info.get("embeddings", [])
        if not embeddings:
            continue
        # cosine similarity (vectors are already L2-normalised)
        sims = [float(np.dot(q_vec, np.array(e))) for e in embeddings]
        # average of top-15 matches for robustness
        avg_sim = float(np.mean(sorted(sims, reverse=True)[:15]))
        if avg_sim > best_sim:
            best_sim = avg_sim
            best_name = info["first_name"]

    if best_sim >= RECOG_THRESHOLD:
        return best_name, best_sim
    return None, best_sim

# ──────────────────────────────────────────────────────────────────────────────
# ATTENDANCE LOGGER
# ──────────────────────────────────────────────────────────────────────────────

def log_attendance(name: str):
    now = datetime.datetime.now()
    with attendance_lock:
        if name not in recognized_names:
            t = now.strftime("%H:%M:%S")
            recognized_names[name] = {
                "time": t,
                "date": now.strftime("%d-%m-%Y"),
                "late": t > DEADLINE,
            }

# ──────────────────────────────────────────────────────────────────────────────
# ATTENDANCE VIDEO STREAM  (Flask → index.html)
# Pipeline:
#   1. yolov8n-face detects all faces in the frame
#   2. Each face crop is embedded and matched against enrolled DB
#   3. Matching name is drawn on the bounding box and logged
# ──────────────────────────────────────────────────────────────────────────────

def generate_frames(camera_id: str):
    global current_camera
    if current_camera is not None:
        current_camera.release()
        current_camera = None

    src = "http://192.168.0.110:6677/video" if camera_id == "ip" else int(camera_id)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open camera: {camera_id}")
        return

    current_camera = cap
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detect all faces in this frame
        boxes = detect_faces(frame)

        for (x1, y1, x2, y2, conf) in boxes:
            face_crop = crop_face(frame, x1, y1, x2, y2)
            if face_crop is None:
                continue

            # Try to recognise every detected face
            q_emb = extract_embedding(face_crop)
            name  = "Unknown"
            sim   = 0.0

            if q_emb is not None:
                q_vec = np.array(q_emb)
                best_name, best_sim = None, -1.0
                with enrollment_lock:
                    for roll, info in enrollment_db.items():
                        embeddings = info.get("embeddings", [])
                        if not embeddings:
                            continue
                        sims = [float(np.dot(q_vec, np.array(e))) for e in embeddings]
                        avg  = float(np.mean(sorted(sims, reverse=True)[:15]))
                        if avg > best_sim:
                            best_sim  = avg
                            best_name = info["first_name"]

                if best_sim >= RECOG_THRESHOLD and best_name:
                    name = best_name
                    sim  = best_sim
                    log_attendance(name)

            # Draw bounding box
            color = (0, 220, 0) if name != "Unknown" else (180, 180, 180)
            label = f"{name}  {sim:.2f}" if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_count += 1
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — Main
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll')
def enroll_page():
    return render_template('enroll.html')

# @app.route('/video_feed/<camera_id>')
# def video_feed(camera_id):
#     return Response(generate_frames(camera_id),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    data = request.json
    frame_b64 = data.get("frame")

    if not frame_b64:
        return jsonify({"success": False}), 400

    img_bytes = base64.b64decode(frame_b64.split(',')[-1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    boxes = detect_faces(frame)
    results = []

    for (x1, y1, x2, y2, conf) in boxes:
        face_crop = crop_face(frame, x1, y1, x2, y2)
        name = "Unknown"
        sim = 0.0

        if face_crop is not None:
            emb = extract_embedding(face_crop)
            if emb:
                q_vec = np.array(emb)
                best_name, best_sim = None, -1.0

                for roll, info in enrollment_db.items():
                    sims = [float(np.dot(q_vec, np.array(e)))
                            for e in info.get("embeddings", [])]
                    if sims:
                        avg = float(np.mean(sorted(sims, reverse=True)[:10]))
                        if avg > best_sim:
                            best_sim = avg
                            best_name = info["first_name"]

                if best_sim >= RECOG_THRESHOLD:
                    name = best_name
                    sim = best_sim
                    log_attendance(name)

        results.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "name": name,
            "sim": sim
        })

    return jsonify({"success": True, "faces": results})

@app.route('/get_attendance')
def get_attendance():
    with attendance_lock:
        return jsonify(recognized_names)

@app.route('/send_attendance', methods=['POST'])
def send_attendance():
    data = request.json
    recipient_email = data.get("email")

    if not recipient_email:
        return {"message": "Email is required."}, 400

    with attendance_lock:
        if not recognized_names:
            return {"message": "No attendees to send."}, 400

        rows = "".join(
            f"<tr><td>{n}</td><td>{i['date']}</td><td>{i['time']}</td>"
            f"<td>{'Yes' if i['late'] else 'No'}</td></tr>"
            for n, i in recognized_names.items()
        )

        html = f"""
        <h2>Attendance Report</h2>
        <table border='1' cellpadding='8' style='border-collapse:collapse'>
        <tr><th>Name</th><th>Date</th><th>Time</th><th>Late</th></tr>
        {rows}
        </table>
        """

    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = recipient_email
        msg['Subject'] = "Attendance Report"

        msg.attach(MIMEText(html, 'html'))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
        server.quit()

        return {"message": f"Attendance email sent to {recipient_email}."}, 200

    except Exception as e:
        print(e)
        return {"message": "Failed to send email."}, 500
@app.route('/export_attendance', methods=['POST'])
def export_attendance():
    with attendance_lock:
        if not recognized_names:
            return {"message": "No attendees to export."}, 400
        df = pd.DataFrame([
            {"Name": n, "Time": i["time"], "Date": i["date"], "Late": i["late"]}
            for n, i in recognized_names.items()
        ])
        fname = f"attendance_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.xlsx"
        df.to_excel(fname, index=False)
        return {"message": f"Saved as {fname}"}, 200

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — Enrollment
# enroll.html captures frames via browser getUserMedia (no Flask camera).
# Frames are sent here as base64 JPEGs; yolov8n-face crops each face
# server-side before embedding and storing.
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/capture_frames', methods=['POST'])
def capture_frames():
    global enrollment_db
    data        = request.json
    first_name  = data.get("first_name", "").strip()
    last_name   = data.get("last_name",  "").strip()
    roll_number = data.get("roll_number","").strip()
    frames_b64  = data.get("frames", [])

    if not first_name or not roll_number:
        return jsonify({"success": False,
                        "message": "First name and roll number are required."}), 400
    if not frames_b64:
        return jsonify({"success": False, "message": "No frames received."}), 400

    embeddings  = []
    saved_count = 0
    person_dir  = ENROLLMENT_DIR / roll_number
    person_dir.mkdir(exist_ok=True)

    for i, b64_str in enumerate(frames_b64):
        try:
            img_bytes = base64.b64decode(b64_str.split(',')[-1])
            nparr     = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # yolov8n-face crops the face from the browser frame
            face_crop, _ = detect_largest_face(frame)
            if face_crop is None:
                continue

            emb = extract_embedding(face_crop)
            if emb is not None:
                embeddings.append(emb)
                saved_count += 1
                # Save every 10th crop for debugging / reference
                if i % 10 == 0:
                    cv2.imwrite(str(person_dir / f"face_{i:03d}.jpg"), face_crop)
        except Exception as e:
            print(f"Frame {i} error: {e}")
            continue

    if saved_count < 5:
        return jsonify({
            "success": False,
            "message": (f"Only {saved_count} valid face frames captured. "
                        "Ensure good lighting and face the camera directly."),
        }), 400

    with enrollment_lock:
        enrollment_db[roll_number] = {
            "first_name":  first_name,
            "last_name":   last_name,
            "roll_number": roll_number,
            "embeddings":  embeddings,
            "enrolled_at": datetime.datetime.now().isoformat(),
        }
        save_enrollment_db(enrollment_db)

    return jsonify({
        "success": True,
        "message": f"Enrolled {first_name} {last_name} with {saved_count} face samples.",
        "count":   saved_count,
    })


@app.route('/get_enrolled')
def get_enrolled():
    global enrollment_db
    with enrollment_lock:
        people = [
            {
                "roll_number": v["roll_number"],
                "first_name":  v["first_name"],
                "last_name":   v["last_name"],
                "enrolled_at": v.get("enrolled_at", ""),
                "samples":     len(v.get("embeddings", [])),
            }
            for v in enrollment_db.values()
        ]
    return jsonify(people)


@app.route('/delete_enrolled/<roll_number>', methods=['DELETE'])
def delete_enrolled(roll_number):
    global enrollment_db
    with enrollment_lock:
        if roll_number not in enrollment_db:
            return jsonify({"success": False, "message": "Not found."}), 404
        del enrollment_db[roll_number]
        save_enrollment_db(enrollment_db)
        p = ENROLLMENT_DIR / roll_number
        if p.exists():
            shutil.rmtree(p)
    return jsonify({"success": True, "message": "Enrollment deleted."})


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)