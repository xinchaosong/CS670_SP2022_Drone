import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score

  
def expand_bounding_box(ystart, xstart, ystop, xstop):
  if ystart - 5 <= 0:
    ystart = 0 
  else:
    ystart -= 5
  if xstart - 5 <= 0:
    xstart = 0  
  else:
    xstart -= 5
  if ystop + 5 >= img.shape[0]:
    ystop = img.shape[0]
  else: 
    ystop += 5
  if xstop + 5 >= img.shape[1]:
    xstop = img.shape[1]
  else:
    xstop += 5
  return ystart, xstart, ystop, xstop

def kmeans_helper(img):
  #Find Grid bondary points
  data = np.nonzero(img)
  topLeft = ( data[1].min(), data[0].min() )
  bottomRight = ( data[1].max(), data[0].max() ) 
  #print("TopLeft: ", topLeft, "\tbottomRight: ", bottomRight)

  plt.scatter(topLeft[0], topLeft[1],c = 'r')
  plt.scatter(bottomRight[0], bottomRight[1], c='r')
  plt.show()

  #creating the grid
  grid_length = 80  #length of each grid square
  grid_height = 80
  centroids = []    #the centroid for each grid
  end_of_row = False
  end_of_col = False
  
  current_x = 0 #coordinates are local to the grid local
  next_x = 0
  current_y = 0
  next_y = 0
  offset_x = topLeft[0]
  offset_y = topLeft[1]
  end_x = bottomRight[0] - topLeft[0]
  end_y = bottomRight[1] - topLeft[1]
  start_grid = time.time()

  #height, width for image array
  while not end_of_col:
    if current_y + grid_height > end_y:
      next_y = current_y + (end_y - current_y) % current_y
      end_of_col = True
    else:
      next_y = current_y + grid_height
    #move across the rows
    while not end_of_row:
      if current_x + grid_length > end_x:
        next_x = current_x + (end_x - current_x) % current_x #current plus remainder
        end_of_row = True
      else:
        next_x = current_x + grid_length
      grid = img[ current_y + offset_y : next_y + offset_y, current_x + offset_x : next_x + offset_x] #get only this grid square from the image
      nonzero_xs, nonzero_ys = np.nonzero(grid)
      #need to bring nonzero pixels into image coordinates
      nonzero_xs += current_x + offset_x
      nonzero_ys += current_y + offset_y
      x_avg = 0
      y_avg = 0
      if len(nonzero_xs) > 0:
        x_avg = np.mean(nonzero_xs)
        y_avg = np.mean(nonzero_ys)
        #print("centroid = ", x_avg, ", ", y_avg)
        centroids.append( [x_avg,y_avg] )
      current_x  = next_x
    #reset x coord
    current_x = 0
    next_x = 0
    #increment y coord
    end_of_row = False
    current_y = next_y

  end_grid = time.time()
  print("Size of grids: ", grid_height, "x", grid_length )
  print("Total Number of centroids after grid algorithm: ",  len(centroids))
  print("Time taken to form grid and calculate centroids: ", end_grid - start_grid)
  centroids = np.asarray(centroids).reshape(-1,2)

  #calculate centroids
  best_centroids = None
  best_labels = None
  best_sil_score = -1
  start_Ctime = time.time() 

  for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(centroids) #all nonzero pixel coords
    silhouette_avg = silhouette_score(centroids, kmeans.labels_)
    #print("For K =", k,
    #      "The average silhouette_score is :", silhouette_avg)
    if silhouette_avg > best_sil_score:
      best_centroids = kmeans.cluster_centers_
      best_labels = kmeans.labels_
      best_sil_score = silhouette_avg
    
  end_Ctime = time.time()
  print("Time taken to run ", k, " kmeans: ", end_Ctime - start_Ctime)

  num_labels = np.unique(best_labels)
  clusters = [centroids[best_labels == label] for label in num_labels]
  return clusters

# # if __name__ == "__main__":
    # # start_time = time.time()
    # # # img = cv2.imread("/content/drive/My Drive/back_sub_binary2.png", cv2.IMREAD_GRAYSCALE)
    # # img = cv2.imread("/content/drive/My Drive/back_sub_binary.png", cv2.IMREAD_GRAYSCALE) 

    # # plt.imshow(img, cmap=plt.cm.gray)
    # # plt.title("Regular image", color='white')
    # # plt.show()

    # # clusters = kmeans_helper(img)

    # # # ------------------ Section is for visuals only
    # # colors = ['c','r']
    # # for c in clusters:
      # # plt.scatter(c[:,0],c[:,1], cmap = 'rainbow')
    # # plt.title("Nonzero Clusters (Grid start,end points in red)", color='white')
    # # plt.xlim(0,1920)
    # # plt.ylim(1080, 0)
    # # plt.show()
    # # # ------------------

    # # for c in clusters:
      # # (xstart, ystart), (xstop, ystop) = c.min(0), c.max(0) 
      # # # expand the bounding box to include feature points | check if bounding box is going off frame
      # # ystart, xstart, ystop, xstop = expand_bounding_box( int(ystart), int(xstart), int(ystop), int(xstop) )
      # # img_mask = cv2.rectangle(img, (xstart, ystart), (xstop, ystop), color = (255,255,255), thickness=-1) 

    # # plt.imshow(img_mask, cmap=plt.cm.gray)
    # # plt.title("New Mask",  color='white')
    # # plt.show()

    # # end_time = time.time()

#print("Time Taken (Including plots) = ", end_time - start_time)


