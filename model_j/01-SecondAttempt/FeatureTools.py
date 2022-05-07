#!/usr/bin/env python

#Use: Library to collect features from images.

#Modules:
#--------------------------------------------------------------------------------------
#common
import numpy as np                              #library for working with arrays
import statistics as stat

#mine
import Imaging


#PACKAGE-STATIC FUNCTIONS: 
#--------------------------------------------------------
def FeatureScan(binary, widthFactor):
    """
        Scans an image and extracts features for use in the reference point model.
    
        @param binary      - cv2 image object   - binary image being analyzed.
        @param widthFactor - string             - indicates direction of analysis (edge to center of image).
        @return widthChar  - list of ints       - list of widths to characterize image in one direction.
        @return totalPix   - list of ints       - num of pixels in [L,R,U,D] directions, considering 'widthFactor'.
        @return numAxPix   - list of ints       - total volume of pixels in [L,R,U,D] directions.
    """
    #intialization
    #---get lengths
    [size_y, size_x] = np.shape(binary)
    x_direction_length = max([2,int((size_x//2)*widthFactor)])
    y_direction_length = max([2,int((size_y//2)*widthFactor)])
    volume_LeftRight   = size_y * x_direction_length
    volume_UpDown      = size_x * y_direction_length
    
    #---get endpoints
    left_end  = x_direction_length
    right_end = size_x - x_direction_length
    up_end    = size_y - y_direction_length
    down_end  = y_direction_length

    #calculate values for all four directions of image
    #---total pixels and numAxialPixels
    totalPix = [volume_LeftRight, volume_LeftRight, volume_UpDown, volume_UpDown]
    numAxPix = [x_direction_length, x_direction_length, y_direction_length,y_direction_length]

    
    #find width for each row or column in image up to and including the image center
    LEFT  = [Imaging.GetFULLRangeOfWhitePixels(binary[:,i]) for i in range(1,left_end)]
    RIGHT = [Imaging.GetFULLRangeOfWhitePixels(binary[:,i]) for i in range(size_x-1, right_end, -1)]
    UP    = [Imaging.GetFULLRangeOfWhitePixels(binary[i,:]) for i in range(size_y-1, up_end,    -1)]
    DOWN  = [Imaging.GetFULLRangeOfWhitePixels(binary[i,:]) for i in range(1,down_end)]

    widths = []
    try:
        widths.append([stop-start if (stop is not -1) else 0 for start,stop in LEFT])
        widths.append([stop-start if (stop is not -1) else 0 for start,stop in RIGHT])
        widths.append([stop-start if (stop is not -1) else 0 for start,stop in UP])
        widths.append([stop-start if (stop is not -1) else 0 for start,stop in DOWN])
    except:
        print("LEFT, RIGHT, UP, DOWN: ", LEFT, RIGHT, UP, DOWN)
        Imaging.showImage(binary)
        raise
    return widths, totalPix, numAxPix 



def CollectFeaturePoints(binary, depth=0.15):
    """
        Obtains all features of an image for use in the reference point model.
        
        @param binary    - cv2 image object - binary image being analyzed.
        @param depth     - float              - fraction of distance between edge of frame and center to perform scan.
        @return features - list of ints       - list of widths to characterize image in one direction.
    """
    #OBTAIN RAW FEATURES
    widths, totalPix, numAxPix  = FeatureScan(binary, depth);
    width_L, width_R, width_U, width_D = widths
    pixel_L, pixel_R, pixel_U, pixel_D = totalPix
    axial_L, axial_R, axial_U, axial_D = numAxPix   
    
    #CALCULATE SUPERIOR FEATURES (FACTORS)
    #---common values
    [mean_L, mean_R, mean_U, mean_D] = list(map(lambda x: stat.mean(x), widths))
    [ max_L,  max_R,  max_U,  max_D] = list(map(lambda x: max(x), widths))
    [wPix_L, wPix_R, wPix_U, wPix_D] = list(map(lambda x: sum(x), widths))
    
    #---stability factors (standard deviation of widths on each edge, normalized by mean value)
    stab_L = np.std(width_L)/max(mean_L,1)
    stab_R = np.std(width_R)/max(mean_R,1)
    stab_U = np.std(width_U)/max(mean_U,1)
    stab_D = np.std(width_D)/max(mean_D,1)

    #---shape factors (max width normalized by num of white pixels)
    shape_L = max_L/max(wPix_L,1)
    shape_R = max_R/max(wPix_R,1)
    shape_U = max_U/max(wPix_U,1)
    shape_D = max_D/max(wPix_D,1)

    #---bounding factors (location of max width relative to edge (fraction))
    bound_L = width_L.index(max_L)/axial_L
    bound_R = width_R.index(max_R)/axial_R
    bound_U = width_U.index(max_U)/axial_U
    bound_D = width_D.index(max_D)/axial_D  

    #---space factors (percentage of white pixels in area)
    space_L = wPix_L/pixel_L
    space_R = wPix_R/pixel_R
    space_U = wPix_U/pixel_U
    space_D = wPix_D/pixel_D               

    #machine learning input data
    features = [
                stab_L,  stab_R,  stab_U,  stab_D,  #stability factors
                shape_L, shape_R, shape_U, shape_D, #shape factors
                bound_L, bound_R, bound_U, bound_D, #bounding factors
                space_L, space_R, space_U, space_D, #space factors
               ]
    return features


#EOF