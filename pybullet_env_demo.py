#!/usr/bin/env python3

import time

import pybullet as p
import pybullet_data

from gym.envs.drone.drone_render_env import drone_urdf_path

box_size = [0.12, 0.23, 0.23]

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setGravity(gravX=0, gravY=0, gravZ=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

p.loadURDF("plane.urdf")
p.loadURDF(str(drone_urdf_path), basePosition=[-0.6, 0, 0.2])

box_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[1, 1, 1])
box_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          halfExtents=box_size,
                                          rgbaColor=[193 / 256, 154 / 256, 108 / 256, 1.0])
box_id = p.createMultiBody(baseCollisionShapeIndex=box_collision_shape_id,
                           baseVisualShapeIndex=box_visual_shape_id,
                           basePosition=[0, 0, box_size[2] + 0.01])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

start_time = time.time()
while time.time() - start_time < 60:
    p.stepSimulation()
