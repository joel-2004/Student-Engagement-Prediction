import cv2
import json
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def save_user_details(user_id, user_name):
    """Save user ID and name mapping in names.json"""
    file_path = "names.json"
    data = {}

    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

    # Add new user or update existing
    data[user_id] = user_name

    # Save updated data
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def create_dataset(user_id, user_name):
    """Capture images and store user details"""
    cap = cv2.VideoCapture(0)
    sample_num = 0

    save_user_details(user_id, user_name)  # Save user info

    if not os.path.exists("dataSet"):
        os.makedirs("dataSet")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"dataSet/User.{user_id}.{sample_num}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.waitKey(100)

        cv2.imshow("Face", frame)
        cv2.waitKey(1)

        if sample_num > 100:  # Capture 100 images
            break

    cap.release()
    cv2.destroyAllWindows()

# Example Usage:
if __name__ == "__main__":
    user_id = input("Enter Student ID: ")
    user_name = input("Enter Student Name: ")
    create_dataset(user_id, user_name)
