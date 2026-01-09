import cv2
import mediapipe as mp
import math
import numpy as np

# --- INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# --- CONSTANTS ---
# Eye Landmarks (MediaPipe FaceMesh)
LEFT_EYE = [33, 133, 160, 144, 158, 153, 145, 154] # Simplified
RIGHT_EYE = [362, 263, 387, 373, 385, 380, 374, 381] # Simplified
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_EYE_TOP_BOTTOM = [159, 145]
L_EYE_LEFT_RIGHT = [33, 133]

class FaceHandController:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        # Flip frame for mirror effect and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape

        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

        return face_results, hand_results, img_w, img_h, frame

    def get_blink_ratio(self, landmarks, eye_indices):
        # Calculate Eye Aspect Ratio (EAR)
        # Vertical distance
        top = landmarks[eye_indices[2]] # 160
        bottom = landmarks[eye_indices[6]] # 145
        v_dist = math.hypot(top.x - bottom.x, top.y - bottom.y)

        # Horizontal distance
        left = landmarks[eye_indices[0]] # 33
        right = landmarks[eye_indices[1]] # 133
        h_dist = math.hypot(left.x - right.x, left.y - right.y)

        if h_dist == 0: return 0
        return v_dist / h_dist

    def detect_hand_gesture(self, hand_landmarks):
        """
        Returns: 'OPEN', 'PEACE', 'THUMBS_UP', or None
        """
        # Finger states (0 = thumb, 1=index, etc.)
        fingers = []
        
        # Tip landmarks: 4, 8, 12, 16, 20
        # PIP landmarks: 2, 6, 10, 14, 18
        
        # Thumb (Check x-axis for thumb roughly) - simplified logic for right/left hand
        if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
            fingers.append(1) # Up
        else:
            fingers.append(0) # Down

        # Other 4 fingers (Check Y axis: Tip < PIP means finger is UP in screen coords)
        for id in [8, 12, 16, 20]:
            if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Logic
        if fingers == [1, 1, 1, 1, 1]:
            return "OPEN_PALM" # Browser
        if fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            return "VICTORY" # Scroll
        if fingers == [1, 0, 0, 0, 0]:
            return "THUMBS_UP" # Media
        
        return None
