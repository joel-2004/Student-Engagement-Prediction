from flask import Flask, flash, redirect, render_template, request, url_for
import cv2
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
import time

app = Flask(__name__)

# Load Face Recognition Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained recognizer model
recognizer_file = "recognizer.yml"
if os.path.exists(recognizer_file):
    recognizer.read(recognizer_file)
else:
    print("Recognizer model not found. Train the model first.")

# Attendance file
attendance_file = "attendance.csv"

# Create CSV if it doesn't exist
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Student_ID", "Student_Name", "Engagement", "Timestamp"])
    df.to_csv(attendance_file, index=False)

# Load student names
names_file = "names.json"
if os.path.exists(names_file):
    with open(names_file, "r") as file:
        name_data = json.load(file)
else:
    name_data = {}


def mark_attendance(student_id, student_name, engagement_level):
    """Mark attendance for detected students, ensuring both ID and Name are recorded."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_date = datetime.today().strftime("%Y-%m-%d")

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
    else:
        df = pd.DataFrame(columns=["Student_ID", "Student_Name", "Engagement", "Timestamp"])

    student_mask = (df["Student_ID"] == str(student_id)) & (df["Timestamp"].str.startswith(today_date))

    if not df[student_mask].empty:
        df.loc[student_mask, "Engagement"] = engagement_level
        df.loc[student_mask, "Timestamp"] = timestamp
    else:
        new_entry = pd.DataFrame([[str(student_id), student_name, engagement_level, timestamp]],
                                 columns=["Student_ID", "Student_Name", "Engagement", "Timestamp"])
        df = pd.concat([df, new_entry], ignore_index=True)

    df.to_csv(attendance_file, index=False)
    print(f"✅ Attendance Updated: {student_name} (ID: {student_id}) - {engagement_level}")




import cv2
import time

def recognize_face():
    """Continuously detect faces, issue warnings, and mark absent if warnings reach 10 within 10 seconds."""
    cap = cv2.VideoCapture(0)  # Open webcam
    recognized_students = set()  # Stores students who were seen at least once
    warning_counts = {}  # Tracks warning count for each student
    warning_timestamps = {}  # Tracks the time of the first warning for each student
    absent_students = set()  # Stores students who reached 10 warnings

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        detected_students = set()
        current_time = time.time()  # Get the current timestamp

        for (x, y, w, h) in faces:
            student_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 50:  # Confidence threshold for recognizing a student
                student_name = name_data.get(str(student_id), "Unknown")
                detected_students.add(student_id)

                # Reset warning count if the student is detected
                warning_counts[student_id] = 0
                warning_timestamps.pop(student_id, None)  # Remove timestamp if student is present

                if student_id not in recognized_students:
                    mark_attendance(student_id, student_name, "Present")
                    recognized_students.add(student_id)

                # Draw a **green** rectangle for recognized faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{student_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            else:
                # **Red frame for unrecognized faces**
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Turn", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print(f"Face not recognized (Confidence: {confidence})")

        # Track students not detected
        for student_id in recognized_students - absent_students:
            if student_id not in detected_students:
                if student_id not in warning_timestamps:
                    warning_timestamps[student_id] = current_time  # Start timer for warnings

                warning_counts[student_id] = warning_counts.get(student_id, 0) + 1
                elapsed_time = current_time - warning_timestamps[student_id]

                print(f"Warning {warning_counts[student_id]} for {name_data.get(str(student_id), 'Unknown')} (Time: {elapsed_time:.2f}s)")

                # **If warning count reaches 10 within 10 seconds, mark absent**
                if warning_counts[student_id] >= 100 :
                    mark_attendance(student_id, name_data.get(str(student_id), "Unknown"), "Absent")
                    absent_students.add(student_id)  # Stop tracking this student
                    print(f"{name_data.get(str(student_id), 'Unknown')} marked absent.")
                    break  # Exit loop if any student is marked absent

        # **Mark missing students with a red frame if warnings exist**
        for (x, y, w, h) in faces:
            if student_id in warning_counts and warning_counts[student_id] > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red frame for warnings
                cv2.putText(frame, f"Warning {warning_counts[student_id]}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to quit

    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect', methods=["POST"])
def detect():
    recognize_face()
    flash("Attendance captured for detected students with high accuracy.", "success")
    return redirect(url_for("index"))

@app.route('/attendance')
def view_attendance():
    """Display attendance records."""
    df = pd.read_csv(attendance_file) if os.path.exists(attendance_file) else pd.DataFrame()
    return render_template("attendance.html", tables=[df.to_html()], titles=df.columns.values) if not df.empty else "No attendance records found."

if __name__ == '__main__':
    app.run(debug=True)
