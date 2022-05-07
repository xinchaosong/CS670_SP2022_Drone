#!/usr/bin/env python3

import time
import threading

import cv2

from tello.tello import Tello
from model_j import decision_nn

switch_on = True
frame = None
text = ''
keymap = {
    'down': 'move down',
    'left': 'move left',
    'forward': 'move forward',
    'no_obstacle': 'no obstacle',
    'right': 'move right',
    'stop': 'drone stop',
    'up': 'move up',
    '': ''
}
preScaleFactor = 0.1
box_recognized = []


def flight_control():
    global text
    global box_recognized

    box_detector = decision_nn.BoxDetector(printBoxesToImages=True, preScaleFactor=0.30,
                                        predictionConfidenceThreshold=0.9, SS_SigmaGaussFilter=1, SS_Scale=500)

    tello.execute_command("speed 10")
    tello.execute_command("takeoff")
    tello.execute_command("down 40")
    time.sleep(1)

    while switch_on:
        action = box_detector.make_decision(frame)
        box_recognized = box_detector.box_recognized
        text = keymap[action]

        if action == 'stop':
            tello.execute_command("land")

        elif action in ['down', 'left', 'forward', 'right', 'up']:
            command = "%s 20" % action
            print(command)
            tello.execute_command(command)

    tello.execute_command("land")


def streaming():
    global switch_on
    global frame

    try:
        while switch_on:
            frame = tello.read()

            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.putText(img=frame,
                            text=text,
                            org=(200, 600),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=3,
                            color=(0, 0, 255),
                            thickness=3,
                            lineType=cv2.LINE_4)

                if box_recognized is not None and len(box_recognized) == 4:
                    x, y, w, h = box_recognized
                    cv2.rectangle(frame, (x, y), (x + w, y + h), [255, 105, 180], 2)

                cv2.imshow('Drone Front Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                switch_on = False

    finally:
        switch_on = False
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tello = Tello()

    flight_control_thread = threading.Thread(target=flight_control)
    flight_control_thread.start()

    streaming()

    flight_control_thread.join()

    tello.close()
