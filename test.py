import dlib
from imutils import face_utils
import cv2
import numpy as np
import time
import winsound

# Constants for eye aspect ratio (EAR) to determine if the eye is closed
EYE_AR_THRESH = 0.27  # Threshold for EAR (below this, eye is considered closed)
EYE_AR_CONSEC_FRAMES = 5 * 30  # Number of consecutive frames the eye must be below the threshold (5 seconds at 30 FPS)

# Initialize counters
COUNTER = 0  # Counts consecutive frames with EAR below threshold

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the Euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])

    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Path to the pre-trained model
model_path = "models/shape_predictor_68_face_landmarks.dat"

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# Start the video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from camera.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # Get the facial landmarks for the detected face
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        left_eye = shape[42:48]  # Points for the left eye
        right_eye = shape[36:42]  # Points for the right eye

        # Calculate the eye aspect ratio (EAR) for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1  # Increment the counter
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                detection_time = time.strftime("%H:%M:%S", time.localtime())
                sound_time = time.strftime("%H:%M:%S", time.localtime(time.time() + 5))
                print(f"Eye closed detected for 5 seconds at {detection_time}!")
                print(f"Sound will be played at {sound_time}")
                winsound.Beep(1000, 1000)  # Beep sound for 1 second
                COUNTER = 0  # Reset the counter after detection
        else:
            COUNTER = 0  # Reset the counter if the eye is open

        # Draw the eye landmarks on the frame
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display the EAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit if 'q' is pressed
    if key == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()