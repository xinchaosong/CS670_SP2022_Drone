#!/usr/bin/env python3

import time
import threading

import cv2

from tello.tello import Tello
from model_s import decision_nn

switch_on = True
frame = None
text = ''
keymap = {
    'down': 'move down',
    'left': 'move left',
    'forward': 'move toward object',
    'no_obstacle': 'no obstacle',
    'right': 'move right',
    'stop': 'drone stop',
    'up': 'move up',
    '': ''
}


def flight_control():
    global text

    tello.execute_command("speed 10")
    tello.execute_command("takeoff")
    time.sleep(1)

    while switch_on:
        action = decision_nn.make_decision(frame)
        text = keymap[action]

        if action == 'stop':
            tello.execute_command("land")
        
        elif action in ['down', 'left', 'forward', 'right', 'up']:
            tello.execute_command("%s 10" % action)

    tello.execute_command("land")


def streaming():
    global switch_on
    global frame

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
            cv2.imshow('Drone Front Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            switch_on = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    tello = Tello()

    flight_control_thread = threading.Thread(target=flight_control)
    flight_control_thread.start()

    streaming()

    flight_control_thread.join()

    tello.close()
