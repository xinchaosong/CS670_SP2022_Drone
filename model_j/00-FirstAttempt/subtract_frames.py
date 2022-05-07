import cv2
import os
import time
from matplotlib import pyplot as plt

# Change this line in order to use different scenario-----------------------
imgs_dir = "/content/drive/My Drive/data/000"
os.chdir(imgs_dir)

start = 70
end = 76

dir_list = sorted(os.listdir())
print(dir_list[start:end] , dir_list[end])
prev_images = [cv2.imread(img) for img in dir_list[start:end]] #puts images 50-56 into a list
new_img = cv2.imread(dir_list[end]) #57th image is new frame
for image in prev_images:
  plt.imshow(image[...,::-1])
  plt.title("Prev Image")
  plt.show()

plt.imshow(new_img[...,::-1])
plt.title("New Image")
plt.show()

start_time = time.time()

#Here down is background subtraction

# This background subtraction function allows for a history of images to be kept to determine the background
# The number of recent images to be kept for computing the foreground is determined by history variable 
backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=210, detectShadows=False)
for image in prev_images:
  backSub.apply(image)

fgMask = backSub.apply(new_img)

end_sub_time = time.time()

foreground_color = cv2.bitwise_and(new_img,new_img,mask = fgMask)

end_bitwise_time = time.time()

plt.imshow(fgMask, cmap=plt.cm.gray)
plt.title("FG Mask")
plt.show()

plt.imshow(foreground_color[...,::-1])
plt.title("FG Mask in color")
plt.show()

#cv2.imwrite("/content/drive/My Drive/back_sub_binary2.png", fgMask)

print("Time taken for background_sub applying color: ", end_bitwise_time - start_time )
print("Time taken for background_sub for binary only: ", end_sub_time - start_time )



