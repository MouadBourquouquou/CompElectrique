import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import threading
from collections import deque
import pyttsx3  # For text-to-speech
from playsound import playsound  # For playing sound files

# Constants
EYE_CLOSE_THRESHOLD = 0.25  # Threshold to detect if eyes are closed
EYE_CLOSE_DURATION = 5  # Time (in seconds) to consider the driver as drowsy
MOUTH_OPEN_THRESHOLD = 0.75  # Threshold to detect if the mouth is open (yawning)
MOUTH_OPEN_DURATION = 3  # Time (in seconds) to consider the driver as yawning
EMOTION_DURATION = 2  # Time (in seconds) to consider an emotion as abnormal
ALERT_COOLDOWN = 5  # Time (in seconds) between alerts
FRAME_SKIP_EMOTION = 5  # Analyze emotion every 5 frames
BAD_EMOTIONS = {"angry", "sad", "fear", "surprise"}  # Emotions to consider as bad

# MediaPipe Face Mesh setup
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_faces=1)

# Landmark indices for eyes and mouth
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Points for the left eye
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Points for the right eye
MOUTH_CORNERS_INDICES = [61, 291]  # Points for the mouth corners
MOUTH_VERTICAL_INDICES = [13, 14]  # Points for the mouth height

# Global variables
eye_close_start_time = None  # Tracks when the eyes first closed
mouth_open_start_time = None  # Tracks when the mouth first opened
emotion_start_time = None  # Tracks when a bad emotion first appeared
last_alert_time = 0  # Tracks the last time an alert was triggered
current_emotion = "neutral"  # Tracks the current emotion
emotion_lock = threading.Lock()  # Ensures safe updates to current_emotion
eye_aspect_history = deque(maxlen=15)  # Stores recent eye aspect ratios

# Initialize the text-to-speech engine
alert_engine = pyttsx3.init()

def speak_alert(message):
    """Speak the alert message aloud."""
    alert_engine.say(message)
    alert_engine.runAndWait()


def calculate_eye_aspect_ratio(eye_points):
    """Calculate how open or closed the eyes are."""
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])  # Distance between top and bottom of the eye
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])  # Another vertical distance
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])  # Distance between the sides of the eye
    return (vertical1 + vertical2) / (2 * horizontal)  # Average eye openness


def calculate_mouth_aspect_ratio(mouth_points):
    """Calculate how open or closed the mouth is."""
    vertical = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[3]))  # Distance between top and bottom of the mouth
    horizontal = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[1]))  # Distance between the sides of the mouth
    return vertical / horizontal  # Mouth openness ratio


def analyze_emotion(face_region):
    """Detect the emotion in the face region using DeepFace."""
    global current_emotion
    try:
        # Analyze the emotion in the face region
        analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
        if analysis and analysis[0]['dominant_emotion']:
            dominant_emotion = analysis[0]['dominant_emotion'].lower()  # Get the dominant emotion
            if dominant_emotion in BAD_EMOTIONS:  # Check if it's a bad emotion
                with emotion_lock:
                    current_emotion = dominant_emotion  # Update the current emotion
            else:
                with emotion_lock:
                    current_emotion = "neutral"  # Reset to neutral if it's not a bad emotion
    except Exception as e:
        print(f"Error in emotion analysis: {e}")  # Print errors for debugging


def get_face_region(landmarks, frame):
    """Get the region of the face for emotion analysis."""
    xs = [lm.x * frame.shape[1] for lm in landmarks.landmark]  # X-coordinates of face landmarks
    ys = [lm.y * frame.shape[0] for lm in landmarks.landmark]  # Y-coordinates of face landmarks
    return (
        int(min(xs)) - 20,  # Top-left X
        int(min(ys)) - 20,  # Top-left Y
        int(max(xs) - min(xs)) + 40,  # Width
        int(max(ys) - min(ys)) + 40  # Height
    )

def play_sound_async(sound_file):
    """Play a sound file in a separate thread."""
    try:
        playsound(sound_file)
    except Exception as e:
        print(f"Error playing sound: {e}")


# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = face_mesh.process(rgb_frame)  # Detect face landmarks

    # Initialize variables
    avg_eye_aspect_ratio = 0.0  # Average eye openness
    mouth_aspect_ratio = 0.0  # Mouth openness
    current_time = time.time()  # Current time for tracking durations
    alert_messages = []  # List of alert messages to display

    if results.multi_face_landmarks:  # If a face is detected
        for face_landmarks in results.multi_face_landmarks:
            # Eye detection
            left_eye = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0])
                                for lm in [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]])
            right_eye = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0])
                                 for lm in [face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES]])

            ear = (calculate_eye_aspect_ratio(left_eye) + calculate_eye_aspect_ratio(right_eye)) / 2
            eye_aspect_history.append(ear)
            avg_eye_aspect_ratio = np.mean(eye_aspect_history)

            if avg_eye_aspect_ratio < EYE_CLOSE_THRESHOLD:  # If eyes are closed
                if eye_close_start_time is None:
                    eye_close_start_time = current_time  # Start tracking eye closure
                elif current_time - eye_close_start_time >= EYE_CLOSE_DURATION:
                    alert_messages.append("DROWSY!")  # Add drowsiness alert
                    alert_messages.append("Systeme Freinage Active")  # Add braking system message
            else:
                eye_close_start_time = None  # Reset if eyes are open

            # Yawn detection
            mouth_corners = [(face_landmarks.landmark[i].x * frame.shape[1],
                             face_landmarks.landmark[i].y * frame.shape[0])
                            for i in MOUTH_CORNERS_INDICES]
            mouth_vertical = [(face_landmarks.landmark[i].x * frame.shape[1],
                              face_landmarks.landmark[i].y * frame.shape[0])
                             for i in MOUTH_VERTICAL_INDICES]

            mouth_aspect_ratio = calculate_mouth_aspect_ratio([*mouth_corners, *mouth_vertical])
            if mouth_aspect_ratio > MOUTH_OPEN_THRESHOLD:  # If mouth is open
                if mouth_open_start_time is None:
                    mouth_open_start_time = current_time  # Start tracking mouth openness
                elif current_time - mouth_open_start_time >= MOUTH_OPEN_DURATION:
                    alert_messages.append("YAWNING!")  # Add yawning alert
            else:
                mouth_open_start_time = None  # Reset if mouth is closed

            # Emotion analysis
            frame_count += 1
            if frame_count % FRAME_SKIP_EMOTION == 0:  # Analyze emotion every few frames
                x, y, w, h = get_face_region(face_landmarks, frame)
                face_region = frame[y:y + h, x:x + w]
                if face_region.size > 0:
                    threading.Thread(target=analyze_emotion, args=(face_region,)).start()

    # Check for bad emotions
    with emotion_lock:
        if current_emotion in BAD_EMOTIONS:  # If a bad emotion is detected
            if emotion_start_time is None:
                emotion_start_time = current_time  # Start tracking emotion duration
            elif current_time - emotion_start_time >= EMOTION_DURATION:
                alert_messages.append(f"{current_emotion.upper()}!")  # Add emotion alert
        else:
            emotion_start_time = None  # Reset if no bad emotion

    # Trigger alerts
    # Replace winsound.Beep with this:
    # Replace the alert triggering logic with this:
    if alert_messages and (current_time - last_alert_time) > ALERT_COOLDOWN:
        alert = "ALERT: " + " ".join(alert_messages)
        print(alert)
        
        # Choose the sound file based on the alert content
        if "Systeme Freinage Active" in alert_messages:
            threading.Thread(target=play_sound_async, args=("./audio/fiiiiiiiq.mp3",)).start()  # Play sound.mp3 for drowsiness
        else:
            threading.Thread(target=play_sound_async, args=("./audio/machiBikhir.mp3",)).start()  # Play sound1.mp3 for other alerts
        
        last_alert_time = current_time

    # Display info on the frame
    cv2.putText(frame, f"EAR: {avg_eye_aspect_ratio:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"MAR: {mouth_aspect_ratio:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if alert_messages:
        cv2.putText(frame, "ALERT!", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        for i, text in enumerate(alert_messages):
            cv2.putText(frame, text, (50, 200 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Driver Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()