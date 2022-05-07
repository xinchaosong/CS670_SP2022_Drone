import os
from pathlib import Path

import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of tensorflow CUDA warnings

import tensorflow as tf
import pickle

class_names = ['down', 'left', 'forward', 'no_obstacle', 'right', 'stop', 'up']
model_path = Path(__file__).resolve().parent.parent / 'models' / 'box_model_0414.pkl'
pickled_model = pickle.load(open(model_path, 'rb'))


def make_decision(frame):
    if frame is None:
        return ""

    frame_resized = cv2.resize(frame, (180, 180))
    img_array = tf.keras.utils.img_to_array(frame_resized)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = pickled_model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    decision = class_names[np.argmax(score)]

    return decision
