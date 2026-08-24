import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

draw_color = (0, 140, 255)
shape_size = 10
shape_type = "flower"
min_dist = 15

def draw_flower(img, center, size, color):
    x, y = center
    petal_r = size // 2
    for angle in range(0, 360, 60):
        rad = math.radians(angle)
        px = int(x + size * math.cos(rad))
        py = int(y + size * math.sin(rad))
        cv2.circle(img, (px, py), petal_r, color, -1)
    cv2.circle(img, (x, y), petal_r, (0, 255, 255), -1)

def draw_star(img, center, size, color):
    x, y = center
    pts = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = size if i % 2 == 0 else size // 2
        px = int(x + r * math.cos(angle))
        py = int(y + r * math.sin(angle))
        pts.append([px, py])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            fingers_up = []
            tips = [4, 8, 12, 16, 20]
            for tip in tips:
                fingers_up.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

            if all(fingers_up):
                prev_x, prev_y = 0, 0
            else:
                dist = math.hypot(x - prev_x, y - prev_y)
                if prev_x == 0 and prev_y == 0 or dist > min_dist:
                    if shape_type == "flower":
                        draw_flower(canvas, (x, y), shape_size, draw_color)
                    else:
                        draw_star(canvas, (x, y), shape_size, draw_color)
                    prev_x, prev_y = x, y

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    combined = cv2.add(frame_bg, canvas)

    cv2.putText(combined, f"Shape: {shape_type} (press f=flower, s=star, c=clear, q=quit)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Air Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('f'):
        shape_type = "flower"
    elif key == ord('s'):
        shape_type = "star"

cap.release()
cv2.destroyAllWindows()