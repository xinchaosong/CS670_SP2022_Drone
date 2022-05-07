import time

import cv2
import numpy as np
import gym
from gym import spaces

from model_j import decision_nn
from tello.tello import Tello


class DroneTelloEnv(gym.Env):
    def __init__(self):
        self.text = ''
        self.text_keymap = {'forward': 'move forward',
                            'back': 'move back',
                            'up': 'move up',
                            'down': 'move down',
                            'left': 'move left',
                            'right': 'move right',
                            'stop': 'landing'}

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)
        self.last_valid_obs = None

        self.box_detector = james_nn.BoxDetector(printBoxesToImages=True, preScaleFactor=0.30,
                                                 predictionConfidenceThreshold=0.9, SS_SigmaGaussFilter=1, SS_Scale=500)

        self.drone_z = 0.4
        self.tello = Tello()

        self.tello.execute_command("speed 10")
        self.tello.execute_command("takeoff")
        self.tello.execute_command("down 40")
        time.sleep(1)

    def step(self, action):
        done = False
        reward = 0

        if action == 0:  # forward
            command = 'back'
        elif action == 1:  # back
            command = 'forward'
        elif action == 2:  # upmjj
            command = 'up'
            self.drone_z += 0.2
        elif action == 3:  # down
            command = 'down'
            self.drone_z -= 0.2
        elif action == 4:  # left
            command = 'left'
        elif action == 5:  # right
            command = 'right'
        else:  # stop
            command = 'stop'

        if self.drone_z < 0.2 or self.drone_z > 1:
            done = True

        self.text = self.text_keymap[command]

        if command == 'stop':
            self.tello.execute_command("land")
            done = True
            reward = 100

        elif command in ['forward', 'back', 'up', 'down', 'left', 'right']:
            self.tello.execute_command("%s 20" % command)

        return self._get_obs(), reward, done, {}

    def reset(self, **kwargs):
        return self._get_obs()

    def render(self, mode="human"):
        pass

    def close(self):
        cv2.destroyAllWindows()
        self.tello.close()

    def _get_obs(self):
        while True:
            frame = self.tello.read()
            if frame is not None:
                break

        self.box_detector.predict(frame)
        box_params = self.box_detector.allBoxes_Final

        if not box_params:
            if self.last_valid_obs is None:
                raise ValueError("Invalid starting position")
            else:
                return self.last_valid_obs

        box_params = box_params[0]
        x, y, w, h = box_params

        while True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(img=frame,
                        text=self.text,
                        org=(200, 600),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3,
                        color=(0, 0, 255),
                        thickness=3,
                        lineType=cv2.LINE_4)

            cv2.rectangle(frame, (x, y), (x + w, y + h), [255, 105, 180], 2)
            cv2.imshow('Drone Front Camera', frame)
            if cv2.waitKey(1):
                break

        obs = np.array([x + 0.5 * w, y + 0.5 * h, w / 960])
        self.last_valid_obs = obs

        return obs
