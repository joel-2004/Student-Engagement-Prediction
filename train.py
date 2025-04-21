import cv2
import numpy as np
import os
import json

# Define paths
dataset_path = "dataset"  # Directory where images are stored
recognizer_file = "recognizer.yml"  # File where trained model is saved
names_file = "names.json"  # File to store mapping of student ID to names

# Initialize recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prepare training data
faces = []
ids = []
name_map = {}

image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]

for image_path in image_paths:
    filename = os.path.basename(image_path)  # Get the filename (e.g., "User.01.1.jpg")
    
    try:
        student_id = int(filename.split(".")[1])  # Extract ID from filename
    except ValueError:
        print(f"Skipping invalid file: {filename}")
        continue  # Skip files that don't match expected format

    name_map[student_id] = filename.split(".")[0]  # Store name without extension
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in detected_faces:
        faces.append(gray[y:y + h, x:x + w])
        ids.append(student_id)

# Train recognizer
if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.save(recognizer_file)
    
    with open(names_file, "w") as f:
        json.dump(name_map, f)
    
    print("Training complete! Model saved.")
else:
    print("No valid faces found for training.")
