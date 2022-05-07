#!/usr/bin/env python3

import os
import numpy as np  # linear algebra

import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of tensorflow CUDA warnings

import tensorflow as tf
import pickle

class_names = ['down', 'left', 'move', 'noobscticle', 'right', 'stop', 'up']
pickled_model = pickle.load(open('model.pkl', 'rb'))
Video_FILE = "video/withobj.mp4"
video = cv2.VideoCapture(Video_FILE)

keymap = {
    'down': 'move down',
    'left': 'move left',
    'move': 'move toward object',
    'noobscticle': 'no obstacle',
    'right': 'move right',
    'stop': 'drone stop',
    'up': 'move up',
}

cap = cv2.VideoCapture('video/withobj.mp4')

while True:

    # Capture frames in the video
    ret, frame = cap.read()

    frame = cv2.resize(frame, (180, 180))
    img_array = tf.keras.utils.img_to_array(frame)
    img_array = tf.expand_dims(frame, 0)  # Create a batch

    predictions = pickled_model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    text = keymap.get(class_names[np.argmax(score)])
    cv2.putText(frame,
                text,
                (10, 50),
                font, 0.5,
                (0, 255, 255),
                2,
                cv2.LINE_4)

    # Display the resulting frame
    imS = cv2.resize(frame, (960, 540))
    cv2.imshow('video', imS)

    # creating 'q' as the quit 
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()
