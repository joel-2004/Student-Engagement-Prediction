import cv2
import os
import json
import numpy as np

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # For alignment

# Ensure dataset directory exists
DATASET_DIR = "dataSet"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def save_user_details(user_id, user_name):
    """Save user ID and name mapping in names.json"""
    file_path = "names.json"
    data = {}

    # Load existing data if available
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

    # Add new user details
    data[user_id] = user_name

    # Save updated data
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def apply_clahe(image):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance face images"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)


def align_face(image, face_coords):
    """Align face using eye detection to improve recognition accuracy."""
    x, y, w, h = face_coords
    face_roi = image[y:y + h, x:x + w]
    
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=6)

    if len(eyes) >= 2:
        # Sort eyes by x-coordinates
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye, right_eye = eyes[:2]
        
        # Compute eye centers
        left_eye_center = (float(left_eye[0] + left_eye[2] // 2), float(left_eye[1] + left_eye[3] // 2))
        right_eye_center = (float(right_eye[0] + right_eye[2] // 2), float(right_eye[1] + right_eye[3] // 2))
        
        # Compute rotation angle
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Convert center to float
        center = (float(w / 2), float(h / 2))
        
        # Apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(face_roi, rotation_matrix, (w, h))
        
        return aligned_face

    return face_roi  # Return original face if eyes are not detected



def create_dataset(user_id, user_name, num_samples=400):
    """Capture images and store user details with enhanced accuracy"""
    cap = cv2.VideoCapture(0)
    sample_num = 0

    save_user_details(user_id, user_name)  # Save user info

    while sample_num < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

        for (x, y, w, h) in faces:
            sample_num += 1

            # Face alignment for better accuracy
            aligned_face = align_face(frame, (x, y, w, h))
            
            # Convert to grayscale and apply CLAHE
            gray_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            enhanced_face = apply_clahe(gray_face)

            # Save images (Original & Preprocessed)
            img_path_gray = f"{DATASET_DIR}/{user_name}.{user_id}.{sample_num}.jpg"
            img_path_color = f"{DATASET_DIR}/{user_name}.{user_id}.{sample_num}_color.jpg"
            cv2.imwrite(img_path_gray, enhanced_face)  # Save enhanced grayscale
            cv2.imwrite(img_path_color, aligned_face)  # Save aligned color image

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {sample_num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.waitKey(100)  # Dynamic delay based on capture progress

        cv2.imshow("Face Capture", frame)
        cv2.waitKey(1)

        if sample_num >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Dataset for {user_name} (ID: {user_id}) collected successfully!")

if __name__ == "__main__":
    user_id = input("Enter User ID: ")
    user_name = input("Enter User Name: ")
    create_dataset(user_id, user_name)
