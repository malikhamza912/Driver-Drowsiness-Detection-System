import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate aspect ratio for eyes and mouth
def aspect_ratio(landmarks, indices):
    A = np.linalg.norm(np.array(landmarks[indices[1]]) - np.array(landmarks[indices[5]]))
    B = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[4]]))
    C = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[3]]))
    return (A + B) / (2.0 * C)

# Drowsiness thresholds
EYE_AR_THRESHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 48
YAWN_AR_THRESHOLD = 0.7

# Landmark indices for eyes and mouth
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14, 87, 317]

# Variables for counting frames
eye_close_count = 0

# Streamlit UI setup
st.title('Driver Drowsiness Detection System')

# Set up the webcam in Streamlit
frame_placeholder = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror-like effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Resize for performance (you can adjust the resolution)
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert frame to RGB (as MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    # Initialize detection messages
    detection_message = ""

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        normalized_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Calculate bounding box for the face
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Draw rectangle around the face
        cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw bounding boxes for eyes
        left_eye_coords = [normalized_landmarks[i] for i in LEFT_EYE]
        right_eye_coords = [normalized_landmarks[i] for i in RIGHT_EYE]

        # Left eye bounding box
        left_eye_x_min = min([coord[0] for coord in left_eye_coords])
        left_eye_x_max = max([coord[0] for coord in left_eye_coords])
        left_eye_y_min = min([coord[1] for coord in left_eye_coords])
        left_eye_y_max = max([coord[1] for coord in left_eye_coords])
        cv2.rectangle(frame_resized, (left_eye_x_min, left_eye_y_min), (left_eye_x_max, left_eye_y_max), (0, 255, 0), 2)

        # Right eye bounding box
        right_eye_x_min = min([coord[0] for coord in right_eye_coords])
        right_eye_x_max = max([coord[0] for coord in right_eye_coords])
        right_eye_y_min = min([coord[1] for coord in right_eye_coords])
        right_eye_y_max = max([coord[1] for coord in right_eye_coords])
        cv2.rectangle(frame_resized, (right_eye_x_min, right_eye_y_min), (right_eye_x_max, right_eye_y_max), (0, 255, 0), 2)

        # Draw bounding box for the mouth
        mouth_coords = [normalized_landmarks[i] for i in MOUTH]
        mouth_x_min = min([coord[0] for coord in mouth_coords])
        mouth_x_max = max([coord[0] for coord in mouth_coords])
        mouth_y_min = min([coord[1] for coord in mouth_coords])
        mouth_y_max = max([coord[1] for coord in mouth_coords])
        cv2.rectangle(frame_resized, (mouth_x_min, mouth_y_min), (mouth_x_max, mouth_y_max), (0, 255, 0), 2)

        # Calculate eye aspect ratios for drowsiness detection
        left_eye_ar = aspect_ratio(normalized_landmarks, LEFT_EYE)
        right_eye_ar = aspect_ratio(normalized_landmarks, RIGHT_EYE)

        # Check if eyes are closed
        if left_eye_ar < EYE_AR_THRESHOLD and right_eye_ar < EYE_AR_THRESHOLD:
            eye_close_count += 1
            detection_message = "You are feeling drowsy!"
            if eye_close_count >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame_resized, detection_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                engine.say("You are feeling drowsy. Please take a break.")
                engine.runAndWait()
                eye_close_count = 0
        else:
            eye_close_count = 0

        # Calculate mouth aspect ratio for yawning detection
        mouth_ar = aspect_ratio(normalized_landmarks, MOUTH)
        if mouth_ar > YAWN_AR_THRESHOLD:
            detection_message = "Yawning Detected!"
            cv2.putText(frame_resized, detection_message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            engine.say("Yawning detected. Please stay alert.")
            engine.runAndWait()

    # Convert the frame to display in Streamlit
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    frame_placeholder.image(frame_resized, channels="RGB", use_column_width=True)

# Release resources
cap.release()
cv2.destroyAllWindows()
