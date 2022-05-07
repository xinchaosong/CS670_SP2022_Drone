#!/usr/bin/env python3

import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from model_j import decision_nn

box_size = [0.23, 0.12, 0.23]

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setGravity(gravX=0, gravY=0, gravZ=0)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

box_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[1, 1, 1])
box_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          halfExtents=box_size,
                                          rgbaColor=[193 / 256, 154 / 256, 108 / 256, 1.0])
box_id = p.createMultiBody(baseCollisionShapeIndex=box_collision_shape_id,
                           baseVisualShapeIndex=box_visual_shape_id,
                           basePosition=[0, 0, box_size[2] + 0.01])

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

projection_matrix_kinova = p.computeProjectionMatrixFOV(
    fov=82.6,
    aspect=1.0,
    nearVal=0.01,
    farVal=100
)

drone_x = -0.2
drone_y = 0.8
drone_z = 0.4

view_matrix = p.computeViewMatrix(
    cameraEyePosition=[drone_x, drone_y, drone_z],
    cameraTargetPosition=[drone_x, drone_y - 1, drone_z],
    cameraUpVector=[0, 0, 1]
)

start_time = time.time()
while time.time() - start_time < 3:
    p.stepSimulation()

_, _, rgb_img, _, _ = p.getCameraImage(
    width=960,
    height=720,
    viewMatrix=view_matrix,
    projectionMatrix=projection_matrix_kinova
)

p.disconnect()

rgb_array = np.array(rgb_img)
rgb_array = rgb_array[:, :, :3]
rgb_array = np.ascontiguousarray(rgb_array, dtype=np.uint8)
box_detector = james_nn.BoxDetector(printBoxesToImages=True, preScaleFactor=0.30,
                                    predictionConfidenceThreshold=0.9, SS_SigmaGaussFilter=1, SS_Scale=500)
box_detector.predict(rgb_array)
rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
for x, y, w, h in box_detector.allBoxes_Final:
    cv2.rectangle(rgb_array, (x, y), (x + w, y + h), [255, 105, 180], 2)

while True:

    cv2.imshow('PyBullet Synthetic Camera', rgb_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('test.png', rgb_array)
        break
