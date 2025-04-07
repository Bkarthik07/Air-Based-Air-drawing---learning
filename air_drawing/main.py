from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

app = Flask(__name__)

# Initialize OpenCV & MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Capture from webcam

# Color choices
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Drawing storage
bpoints, gpoints, rpoints, ypoints = [deque(maxlen=1024)], [deque(maxlen=1024)], [deque(maxlen=1024)], [deque(maxlen=1024)]
blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0


def generate_frames():
    global blue_index, green_index, red_index, yellow_index, colorIndex

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        # Draw UI buttons
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx, lmy = int(lm.x * 640), int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger, thumb = (landmarks[8][0], landmarks[8][1]), (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

            # Clear drawing if Clear button is pressed
            if fore_finger[1] <= 65:
                if 40 <= fore_finger[0] <= 140:  # Clear Button
                    bpoints.clear()
                    gpoints.clear()
                    rpoints.clear()
                    ypoints.clear()

                    bpoints.append(deque(maxlen=512))
                    gpoints.append(deque(maxlen=512))
                    rpoints.append(deque(maxlen=512))
                    ypoints.append(deque(maxlen=512))

                    blue_index = green_index = red_index = yellow_index = 0

                elif 160 <= fore_finger[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= fore_finger[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= fore_finger[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= fore_finger[0] <= 600:
                    colorIndex = 3  # Yellow

            elif (thumb[1] - fore_finger[1] < 30):  # New stroke when thumb and index are close
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(fore_finger)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(fore_finger)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(fore_finger)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(fore_finger)

        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        # Draw the stored points
        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Encode frame to JPEG and return as byte stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Load HTML page


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
