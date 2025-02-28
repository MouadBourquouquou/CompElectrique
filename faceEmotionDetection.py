import cv2
import mediapipe as mp
from deepface import DeepFace

# Initialize Mediapipe face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB (Mediapipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                # Analyze emotion using DeepFace
                try:
                    analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    print(f"DeepFace Error: {e}")

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display output
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
