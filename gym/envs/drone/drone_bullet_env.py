import numpy as np
import pybullet as p

import gym
from model_j import decision_nn
from gym import spaces

box_size = (0.23, 0.12, 0.23)


class DroneBulletEnv(gym.Env):
    def __init__(self):

        self.physics_client = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.physics_client)
        p.setGravity(gravX=0, gravY=0, gravZ=0, physicsClientId=self.physics_client)

        box_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                        halfExtents=[1, 1, 1],
                                                        physicsClientId=self.physics_client)
        box_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=box_size,
                                                  rgbaColor=[0.7539, 0.6016, 0.4219, 1.0],
                                                  physicsClientId=self.physics_client)
        p.createMultiBody(baseCollisionShapeIndex=box_collision_shape_id,
                          baseVisualShapeIndex=box_visual_shape_id,
                          basePosition=[0, 0, box_size[2] + 0.01],
                          physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.physics_client)

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=82.6,
            aspect=1.0,
            nearVal=0.01,
            farVal=100,
            physicsClientId=self.physics_client
        )

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)
        self.box_detector = james_nn.BoxDetector(printBoxesToImages=True, preScaleFactor=0.30,
                                                 predictionConfidenceThreshold=0.9, SS_SigmaGaussFilter=1, SS_Scale=500)

        self.drone_x = 0
        self.drone_y = 0
        self.drone_z = 0

        self.last_valid_obs = np.zeros(3)

    def step(self, action):
        done = False

        if action == 0:  # forward
            self.drone_y += 0.2
        elif action == 1:  # back
            self.drone_y -= 0.2
        elif action == 2:  # up
            self.drone_z += 0.2
        elif action == 3:  # down
            self.drone_z -= 0.2
        elif action == 4:  # left
            self.drone_x -= 0.2
        elif action == 5:  # right
            self.drone_x += 0.2
        else:  # stop
            self.drone_z = 0

        obs = self._get_obs()

        reward = 1 - np.linalg.norm(np.array((self.drone_x, self.drone_y)) - np.array((0, 0.5)))

        if self.drone_x < -0.8 or self.drone_x > 0.8 \
                or self.drone_y < 0.4 or self.drone_y > 2 \
                or self.drone_z < 0.2 or self.drone_z > 0.8 \
                or action == 6:
            done = True
            reward = -100

        if -0.2 <= self.drone_x <= 0.2 \
                and 0.4 <= self.drone_y <= 0.6 \
                and 0.2 <= self.drone_z <= 1 \
                and action == 6:
            done = True
            reward = 100

        return obs, reward, done, {'position': (self.drone_x, self.drone_y, self.drone_z)}

    def reset(self, **kwargs):
        while True:
            self.drone_x = -0.4 + np.random.random() * 0.8
            self.drone_y = 0.8 + np.random.random() * 0.8
            self.drone_z = 0.4

            obs = self._get_obs()

            if self.box_detector.allBoxes_Final:
                return obs

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.physics_client)

    def _get_obs(self):
        p.stepSimulation(physicsClientId=self.physics_client)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[self.drone_x, self.drone_y, self.drone_z],
            cameraTargetPosition=[self.drone_x, self.drone_y - 1, self.drone_z],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.physics_client
        )

        _, _, rgb_img, _, _ = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            physicsClientId=self.physics_client
        )

        rgb_array = np.array(rgb_img)[:, :, :3]
        rgb_array = np.ascontiguousarray(rgb_array, dtype=np.uint8)

        self.box_detector.predict(rgb_array)

        box_params = self.box_detector.allBoxes_Final

        if not box_params:
            return self.last_valid_obs

        box_params = box_params[0]
        x, y, w, h = box_params

        obs = np.array([x + 0.5 * w, y + 0.5 * h, w / 960])
        self.last_valid_obs = obs

        return obs
