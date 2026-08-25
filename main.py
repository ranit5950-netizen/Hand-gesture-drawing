import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# Canvas to draw on (same size as webcam frame)
canvas = None

# Store previous fingertip position
prev_x, prev_y = 0, 0

# Drawing color and thickness
draw_color = (255, 0, 255)  # pink
thickness = 8

def fingers_up(hand_landmarks):
    """Returns a list [thumb, index, middle, ring, pinky] -> True if finger is up"""
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (compare x, since it moves sideways)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(True)
    else:
        fingers.append(False)

    # Other 4 fingers (compare y, tip above joint = up)
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

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers = fingers_up(hand_landmarks)

            # Index fingertip position
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Drawing mode: only index finger up
            if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, thickness)
                prev_x, prev_y = x, y
                cv2.circle(frame, (x, y), 10, draw_color, cv2.FILLED)

            # All fingers up = stop drawing (lift pen)
            elif all(fingers):
                prev_x, prev_y = 0, 0

            # Fist = clear canvas
            elif not any(fingers):
                canvas = np.zeros((h, w, 3), np.uint8)
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge canvas onto the frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, mask)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Hand Gesture Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()