#!/usr/bin/env python3

import cv2

video = cv2.VideoCapture("video/withobj.mp4")
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)

# frame
currentframe = 0

while True:

    # reading from frame
    ret, frame = video.read()
    print(currentframe)
    if currentframe % 10 != 0:
        currentframe += 1
        continue
    if ret:
        # if video is still left continue creating images
        name = './data/train/0/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created

    else:
        break
    currentframe += 1

# Release all space and windows once done
video.release()
cv2.destroyAllWindows()
