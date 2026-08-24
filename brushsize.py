import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
thickness = 8

colors = [
    ("Pink", (255, 0, 255), 0, 100),
    ("Blue", (255, 0, 0), 100, 200),
    ("Green", (0, 255, 0), 200, 300),
    ("Red", (0, 0, 255), 300, 400),
    ("Eraser", (0, 0, 0), 400, 500),
]
draw_color = colors[0][1]
current_mode = "Idle"

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(True)
    else:
        fingers.append(False)
    for tip_id in tips[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    for name, color, x1, x2 in colors:
        cv2.rectangle(frame, (x1, 0), (x2, 80), color if name != "Eraser" else (50, 50, 50), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers = fingers_up(hand_landmarks)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Pinch distance between thumb and index = brush size control
            pinch_dist = math.hypot(x - tx, y - ty)

            if y < 80:
                current_mode = "Selecting Color"
                for name, color, x1, x2 in colors:
                    if x1 < x < x2:
                        draw_color = color if name != "Eraser" else (0, 0, 0)
                prev_x, prev_y = 0, 0

            # Pinch mode: thumb + index both up, others down = resize brush
            elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                current_mode = "Resizing Brush"
                thickness = int(np.interp(pinch_dist, [20, 200], [2, 50]))
                cv2.line(frame, (tx, ty), (x, y), (0, 255, 255), 2)
                cv2.putText(frame, f"Size: {thickness}", (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                prev_x, prev_y = 0, 0

            elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                current_mode = "Drawing"
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, thickness)
                prev_x, prev_y = x, y
                cv2.circle(frame, (x, y), thickness // 2 + 2, draw_color, cv2.FILLED)

            elif all(fingers):
                current_mode = "Idle"
                prev_x, prev_y = 0, 0

            elif not any(fingers):
                current_mode = "Cleared"
                canvas = np.zeros((h, w, 3), np.uint8)
                prev_x, prev_y = 0, 0

            else:
                current_mode = "Idle"
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        current_mode = "No Hand Detected"
        prev_x, prev_y = 0, 0

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, mask)
    frame = cv2.bitwise_or(frame, canvas)

    # Status text at bottom
    cv2.putText(frame, f"Mode: {current_mode}  |  Brush Size: {thickness}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Drawing saved as drawing.png")

cap.release()
cv2.destroyAllWindows()