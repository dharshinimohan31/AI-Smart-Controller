import streamlit as st
import cv2
import pyautogui
import numpy as np
import time
import mediapipe as mp
import webbrowser
from gesture_utils import FaceHandController, LEFT_EYE

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Eye & Hand Controller",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #4A90E2;
    text-align: center;
    font-weight: bold;
}
.sub-header {
    font-size: 1.2rem;
    color: #aaa;
    text-align: center;
    margin-bottom: 20px;
}
.status-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1E1E1E;
    border: 1px solid #333;
    text-align: center;
}
.status-active { color: #00FF00; font-weight: bold; }
.status-inactive { color: #FF0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- INIT STATE ----------------
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    mouse_smooth = st.slider("Cursor Smoothing", 1, 10, 5)
    sens_x = st.slider("Horizontal Sensitivity", 1.0, 5.0, 2.5)
    sens_y = st.slider("Vertical Sensitivity", 1.0, 5.0, 2.5)

    enable_cursor = st.checkbox("Enable Eye Cursor", True)
    enable_click = st.checkbox("Enable Blink Click", True)
    enable_gestures = st.checkbox("Enable Hand Gestures", True)

# ---------------- MAIN UI ----------------
st.markdown('<p class="main-header">AI Touchless Interface</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Control your computer with Eyes & Hands</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

# ---------------- CORE SETUP ----------------
screen_w, screen_h = pyautogui.size()
EDGE_MARGIN = 25

controller = FaceHandController()

prev_screen_x, prev_screen_y = screen_w // 2, screen_h // 2
click_cooldown = 0
gesture_cooldown = 0

# ---------------- MAIN FUNCTION ----------------
def run_app():
    global prev_screen_x, prev_screen_y, click_cooldown, gesture_cooldown

    cap = cv2.VideoCapture(0)

    with col1:
        video_placeholder = st.empty()

    with col2:
        status_cursor = st.empty()
        status_blink = st.empty()
        status_gesture = st.empty()

    while st.session_state['camera_active']:
        ret, frame = cap.read()
        if not ret:
            break

        face_results, hand_results, img_w, img_h, frame = controller.process_frame(frame)
        action_text = "Idle"

        # -------- EYE TRACKING --------
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            blink_ratio = controller.get_blink_ratio(landmarks, LEFT_EYE)
            if enable_click and blink_ratio < 0.26:
                now = time.time()
                if now - click_cooldown > 0.5:
                    pyautogui.click()
                    click_cooldown = now
                    action_text = "Blink Click"
                    status_blink.markdown(
                        'Blink: <span class="status-active">DETECTED</span>',
                        unsafe_allow_html=True
                    )
            else:
                status_blink.markdown(
                    'Blink: <span class="status-inactive">WAITING</span>',
                    unsafe_allow_html=True
                )

            if enable_cursor:
                iris_x = landmarks[473].x
                iris_y = landmarks[473].y

                target_x = np.interp(iris_x, [0.43, 0.57], [0, screen_w * sens_x])
                target_y = np.interp(iris_y, [0.43, 0.53], [0, screen_h * sens_y])

                curr_x = prev_screen_x + (target_x - prev_screen_x) / mouse_smooth
                curr_y = prev_screen_y + (target_y - prev_screen_y) / mouse_smooth

                safe_x = max(EDGE_MARGIN, min(screen_w - EDGE_MARGIN, int(curr_x)))
                safe_y = max(EDGE_MARGIN, min(screen_h - EDGE_MARGIN, int(curr_y)))

                pyautogui.moveTo(safe_x, safe_y, duration=0.05)
                prev_screen_x, prev_screen_y = safe_x, safe_y

                status_cursor.markdown(
                    'Cursor: <span class="status-active">TRACKING</span>',
                    unsafe_allow_html=True
                )
            else:
                status_cursor.markdown(
                    'Cursor: <span class="status-inactive">OFF</span>',
                    unsafe_allow_html=True
                )

        # -------- HAND GESTURES --------
        if hand_results.multi_hand_landmarks and enable_gestures:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture = controller.detect_hand_gesture(hand_landmarks)
                now = time.time()

                if gesture and now - gesture_cooldown > 1.0:
                    if gesture == "OPEN_PALM":
                        action_text = "Opening Browser"
                        webbrowser.open("https://www.google.com")
                        gesture_cooldown = now

                    elif gesture == "VICTORY":
                        action_text = "Scrolling"
                        pyautogui.scroll(-300)
                        gesture_cooldown = now + 0.2

                    elif gesture == "THUMBS_UP":
                        action_text = "Play / Pause"
                        try:
                            pyautogui.press('playpause')
                        except pyautogui.FailSafeException:
                            pass
                        gesture_cooldown = now

                    elif gesture == "FIST":
                        action_text = "Emergency Stop"
                        st.session_state['camera_active'] = False
                        break

                    status_gesture.markdown(
                        f'Gesture: <span class="status-active">{gesture}</span>',
                        unsafe_allow_html=True
                    )
                else:
                    status_gesture.markdown(
                        'Gesture: <span class="status-inactive">NONE</span>',
                        unsafe_allow_html=True
                    )

        cv2.putText(
            frame, f"Action: {action_text}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        video_placeholder.image(frame, channels="RGB", use_container_width=True)

    cap.release()

# ---------------- BUTTONS ----------------
with col1:
    if st.button("▶ START CAMERA"):
        st.session_state['camera_active'] = True
        run_app()

    if st.button("⏹ STOP CAMERA"):
        st.session_state['camera_active'] = False
        st.rerun()
