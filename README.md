# AttendX — AI Face Recognition Attendance System

AttendX is a real-time AI-powered face recognition attendance system built using **YOLOv8 Face Detection**, **Flask**, **OpenCV**, and a custom lightweight embedding pipeline.

It supports:

* Real-time browser-based face recognition
* Multi-sample face enrollment
* Cosine similarity matching
* Attendance logging with late detection
* Excel export
* Email report sending (Gmail SMTP)
* Enrolled face management

Designed for deployment on local machines or cloud platforms like Hugging Face Spaces.

---

# Core Architecture

Browser Camera →
YOLOv8 Face Detection →
Custom Embedding Extraction →
Cosine Similarity Matching →
Attendance Logging →
Email / Excel Export

---

# Key Features

## 1. Real-Time Face Detection (YOLOv8)

* Model: `yolov8n-face-lindevs.pt`
* Confidence threshold: 0.50
* Detects all faces in frame
* Largest-face logic for enrollment
* Bounding box + similarity score display

YOLO config directory is redirected for cloud environments:

```python
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
```

---

## 2. Lightweight Face Embedding (No Heavy Recognition Model)

Instead of using FaceNet or ArcFace, the system:

* Resizes face to 128×128
* Converts to grayscale
* Applies histogram equalization
* Divides into 16 blocks
* Computes 32-bin histogram per block
* Concatenates features
* L2-normalizes vector

Fast. CPU-friendly. No GPU required.

---

## 3. Face Recognition

* Cosine similarity matching
* Average of top 10–15 similarities for robustness
* Recognition threshold: 0.72
* Thread-safe enrollment comparison

If similarity ≥ threshold:

* Name displayed
* Attendance logged
* Bounding box turns green

Else:

* Marked as "Unknown"

---

## 4. Smart Enrollment System

Enrollment flow:

1. User fills:

   * First Name
   * Last Name
   * Roll Number

2. Browser captures ~180 frames over ~10 seconds

3. Server:

   * Detects largest face
   * Extracts embeddings
   * Stores multiple embeddings per person
   * Saves every 10th cropped face image

Minimum required valid samples: 5

Enrollment stored in:

```
enrolled_faces/
    ├── <roll_number>/
    └── enrollment_db.json
```

---

## 5. Attendance Logging

Attendance is stored in-memory:

```python
recognized_names = {
    "John": {
        "time": "10:15:22",
        "date": "20-02-2026",
        "late": True/False
    }
}
```

Late detection:

```
DEADLINE = "12:00"
```

If arrival time > deadline → marked Late.

Thread-safe using `Lock()`.

---

## 6. Email Report (Gmail SMTP)

Sends HTML attendance table to recipient.

Uses:

* `smtplib`
* `MIMEMultipart`
* TLS (smtp.gmail.com:587)

Required environment variables:

```
GMAIL_USER
GMAIL_PASS
```

---

## 7. Excel Export

Exports attendance to:

```
attendance_YYYY-MM-DD_HH-MM.xlsx
```


## 8. Enrolled Face Management

Endpoints:

* View enrolled users
* Delete enrolled user
* Automatic directory cleanup

---

# API Endpoints

## Main

| Method | Endpoint             | Description                        |
| ------ | -------------------- | ---------------------------------- |
| GET    | `/`                  | Attendance dashboard               |
| POST   | `/recognize_frame`   | Recognize faces from browser frame |
| GET    | `/get_attendance`    | Fetch attendance data              |
| POST   | `/send_attendance`   | Send email report                  |
| POST   | `/export_attendance` | Export XLSX file                   |

---

## Enrollment

| Method | Endpoint                         | Description         |
| ------ | -------------------------------- | ------------------- |
| GET    | `/enroll`                        | Enrollment page     |
| POST   | `/capture_frames`                | Enroll face samples |
| GET    | `/get_enrolled`                  | List enrolled users |
| DELETE | `/delete_enrolled/<roll_number>` | Delete user         |

---

# Project Structure

```
├── app.py
├── yolov8n-face-lindevs.pt
├── enrolled_faces/
│   ├── enrollment_db.json
│   └── <roll_number>/
├── templates/
│   ├── index.html
│   └── enroll.html
└── requirements.txt
```



---

# Installation

## 1. Clone

```bash
git clone https://github.com/sameermujahid/attendance_application
cd attendance_application
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Add YOLO Face Model

Place:

```
yolov8n-face-lindevs.pt
```

in root directory.

---


## 4. Run

```bash
python app.py
```

Server runs at:

```
http://localhost:7860
```

If deploying to cloud:

* Use `/tmp/Ultralytics` for YOLO config
* Ensure writable directory for `enrolled_faces`
