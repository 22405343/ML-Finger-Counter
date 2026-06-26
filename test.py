import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup MediaPipe Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1, 
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# MediaPipe IDs for Tips and Knuckles
tip_ids = [8, 12, 16, 20]
pip_ids = [6, 10, 14, 18]

cap = cv2.VideoCapture(0)
print("Running Finger Counter... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    total_fingers = 0

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            fingers = []

            # --- SMART THUMB DETECTION ---
            # Determine if it's a left or right hand layout by checking if the 
            # thumb base (2) is to the right or left of the pinky base (17)
            if hand_landmarks[2].x > hand_landmarks[17].x:
                # If True, the thumb needs to be further right to be "open"
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # If False, the thumb needs to be further left to be "open"
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # --- OTHER 4 FINGERS (Vertical check) ---
            for i in range(4):
                if hand_landmarks[tip_ids[i]].y < hand_landmarks[pip_ids[i]].y:
                    fingers.append(1) 
                else:
                    fingers.append(0) 

            total_fingers = fingers.count(1)

            # Draw dots on screen
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    # Display the finger count on the screen
    cv2.putText(
        frame, 
        f"Fingers: {total_fingers}", 
        (40, 90), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        2, 
        (255, 0, 0), 
        3
    )

    cv2.imshow("Finger Counter", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()