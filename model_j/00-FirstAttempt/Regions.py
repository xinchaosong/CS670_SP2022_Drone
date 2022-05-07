#!/usr/bin/env python

#Use: Library to perform actions to find certain regions of an image.

#Modules:
#--------------------------------------------------------------------------------------
import numpy as np                 #library for working with arrays
import cv2                         #libary to solve computer vision problems
import random                      #library for randomization tools
import time                        #library for timing tools
import copy
# import PreprocessingTools as Func  #common tool location
# import matplotlib.pyplot as plt                 #libary for plotting (extension of numpy)
from sklearn.cluster import KMeans,MiniBatchKMeans              #kmeans clustering
from sklearn.metrics import silhouette_score    #silhouette library (for kmeans)
from image_tools.sizes import resize_and_crop   #ability to crop from middle

#mine
import Imaging


#PACKAGE-STATIC FUNCTIONS:
#--------------------------------------------------------------------------------------
def GetMovementRegions(movementMask):
    """
       Obtains most appropriate number (and coordinates of) clusters to find separate hands. 
       Note: The search grid that is established is based on the white pixels in the movement mask. 
             The search grid will be as small as necessary to encapsulate all white pixels, which 
             means that the resolution of the movementMask is moot. 
             
       @param movementMask     - CustomImage object - image object which will be scanned for movement.
       @return success         - bool - indicates if movement regions were able to be found.
       @return movementRegions - list - list of boxes which will capture movement regions.
    """   
    #initialization
    #---flags
    success = True
    
    #---silhouette values
    bestLabels = None
    bestSilhouetteScore = -1

    #---search grid
    gridPartitionsX = 22         #no. of grid partitions in x dimension
    gridPartitionsY = 22         #no. of grid partitions in y dimension
    partitionCenters = []        #center of mass of all objects in each grid partition
    
    #calculate reference points for grid search
    data = np.nonzero(movementMask.image)
    topLeft = [data[1].min(), data[0].min()]
    bottomRight = [data[1].max(), data[0].max()]    
    lowerOffsetX, lowerOffsetY = topLeft
    upperOffsetX, upperOffsetY = bottomRight
    objectRegionX = upperOffsetX - lowerOffsetX
    objectRegionY = upperOffsetY - lowerOffsetY
    
    #calculate boundaries of each grid partition
    searchX = np.linspace(lowerOffsetX, upperOffsetX, num=gridPartitionsX,dtype=int, endpoint=True)
    searchY = np.linspace(lowerOffsetY, upperOffsetY, num=gridPartitionsY,dtype=int, endpoint=True)
    
    #perform grid search
    for i in range(len(searchX)-1):
        for j in range(len(searchY)-1):
            #isolate grid slice of movementMask
            grid = movementMask.image[(searchY[j]):(searchY[j+1]), (searchX[i]):(searchX[i+1])]

            #find non-zero pixels and bring them into movementMask coordinates
            nonzeroX, nonzeroY = np.nonzero(grid)
            nonzeroX += searchX[i]
            nonzeroY += searchY[j]
            if len(nonzeroY) == 1 and len(nonzeroX) == 1:
                partitionCenters.append([int(nonzeroX),int(nonzeroY)]) 
            elif len(nonzeroX) > 1:
                partitionCenters.append([int(np.mean(nonzeroX)), int(np.mean(nonzeroY))])
    partitionCenters = np.asarray(partitionCenters).reshape(-1,2)
    if len(partitionCenters) < 2:
        return False, None, None

    #estimate clusters by contours and do kmeans clustering
    try:
        clusterProposal = max(2,len(cv2.findContours(movementMask.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]))
        kmeansModel = MiniBatchKMeans(n_clusters=clusterProposal, random_state=0, batch_size=256, max_iter=10)
        kmeansModel.fit(partitionCenters)
        clusters = [partitionCenters[kmeansModel.labels_ == label] for label in np.unique(kmeansModel.labels_)]           
    except:
        raise Exception("You're attempting to use more clusters than there are partition COM available! Check Binarization. Check for a noisy image.")

        
#     try:
#         clusterProposal = len(cv2.findContours(movementMask.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
#         if clusterProposal > 1:
#             kmeansModel = MiniBatchKMeans(n_clusters=clusterProposal, random_state=0, batch_size=6, max_iter=10)
#             kmeansModel.fit(partitionCenters)
#             clusters = [partitionCenters[kmeansModel.labels_ == label] for label in np.unique(kmeansModel.labels_)]  
#         else:
#             clusters = [partitionCenters]
#     except:
#         raise Exception("You're attempting to use more clusters than there are partition COM available! Check Binarization. Check for a noisy image.")
        
        
#     #oldMethod
#     try:
#         clusterProposal = len(cv2.findContours(movementMask.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
#         if clusterProposal > 1:
#             kmeansArray = [MiniBatchKMeans(n_clusters=x, random_state=0, batch_size=6, max_iter=10) for x in [2,clusterProposal]]
#             [x.fit(partitionCenters) for x in kmeansArray] 
#             silhouetteAvgScore = np.array([silhouette_score(partitionCenters, x.labels_) for x in kmeansArray])
#             best = np.where(silhouetteAvgScore == silhouetteAvgScore.max())[0][0]
#             bestLabels, bestSilhouetteScore = kmeansArray[best].labels_, silhouetteAvgScore[best]
#             clusters = [partitionCenters[bestLabels == label] for label in np.unique(bestLabels)]
#         else:
#             kmeansModel = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6, max_iter=10)
#             kmeansModel.fit(partitionCenters)
#             clusters = [partitionCenters[kmeansModel.labels_ == label] for label in np.unique(kmeansModel.labels_)]           
#     except:
#         raise Exception("Make sure the image is binarized! It will give incorrect 'clusterProposal' if not binarized.")        
        
        
    #obtain bounding box for each cluster
    movementRegions = []
    for c in clusters:
        #find bounding values of cluster (min/max values) and store values into arrays
        xStart, yStart = c.min(0)
        xStop,  yStop  = c.max(0)
#         if not yStart == yStop and not xStart == xStop and yStop <= movementMask.shape[0] and xStop <= movementMask.shape[1]:
        movementRegions.append([xStart, yStart, xStop, yStop])
    return success, movementRegions

def GetRegionProposal(binary, region):
    """
       Starts in the middle of the hand/arm, and expands a box until it contains the entire hand.
    
       @param binary - CustomImage object - image object which will referenced to find regions for proposal.
       @return       - list of ints - list of form "[x0, y0, x1, y1]" defining a region proposal.
    """
    #use distance transform to find good point inside of region
    #note: distance transform guarantees to find point inside white region.
    bufferImage = copy.deepcopy(binary)
    bufferImage.cropImage(region[0], region[2], region[1], region[3])
    centerPoints = bufferImage.getThresholdedDistanceTransform()
    centerOfMass = centerPoints[0]
    centerOfMass[0] += region[0]
    centerOfMass[1] += region[1]
    
    # initialize     
    readyToStop = xLeftDone = xRightDone = yLeftDone = yRightDone = False
    xLeft, xRight, yLeft, yRight = centerOfMass[0], centerOfMass[0], centerOfMass[1], centerOfMass[1]
    xMax = np.shape(binary)[1]-1
    yMax = np.shape(binary)[0]-1  
    
    #expand box until all white object captured
    while not readyToStop:
        #capture previous iteration's values
        lastXLeft, lastXRight, lastYLeft, lastYRight = xLeft, xRight, yLeft, yRight
        
        #expand box
        #---x direction
        xLeft  -= 1 if xLeft > 0     else 0
        xRight += 1 if xRight < xMax else 0
        
        #---y direction
        yLeft -= 1  if yLeft > 0     else 0
        yRight += 1 if yRight < yMax else 0    

        #find if whitespace still exists
        #note: origin for image objects made by CV2 is in top left corner. So
        #      the yLeft is smaller than yRight.
        left_vertical     = np.count_nonzero(binary.image[yLeft:yRight,xLeft])
        right_vertical    = np.count_nonzero(binary.image[yLeft:yRight,xRight])
        top_horizontal    = np.count_nonzero(binary.image[yLeft,xLeft:xRight])
        bottom_horizontal = np.count_nonzero(binary.image[yRight,xLeft:xRight])           

        #check if ready to stop
        if (xLeft == lastXLeft) or (left_vertical == 0):
            xLeftDone = True
        if (xRight == lastXRight) or (right_vertical == 0):
            xRightDone = True
        if (yLeft == lastYLeft) or (top_horizontal == 0):
            yLeftDone = True
        if (yRight == lastYRight) or (bottom_horizontal == 0):
            yRightDone = True
        readyToStop = xLeftDone and xRightDone and yLeftDone and yRightDone
    return [xLeft, yLeft, xRight, yRight]

def GetRegionsOfInterest(binary, boxes, IOU_Allowance=0.75):
    """
       Uses intersection over union to determine unique region proposals (regions of interest)
       This function should only be used if there is more than one image to consider.
        
       @param binary         - CustomImage object - image object which will referenced to find regions for proposal.
       @param boxes          - list of lists      - list of all region proposals of the form "[x0, y0, x1, y1]".
       @param IOU_Allowance  - float (fraction)   - fraction (percentage) of area of overlap required to match in IOU.
       @return squeezedBoxes - list of lists      - list of all ROI of the form "[x0, y0, x1, y1]".
   """   
    #initialize
    squeezedBoxes = regionsOfInterest = []
    endPoint = len(boxes)-1       
    bufferImage = copy.deepcopy(binary)
    
    #return if boxes is None
    if boxes is None:
        return
    
    #squeeze image for each region proposal box
    for box in boxes:
        #grab an image from list
        bufferImage = copy.deepcopy(binary)
        (actualXStart, actualYStart, actualXStop, actualYStop) = box
        
        #squeeze image to get just hand; obtain offset
        bufferImage.cropImage(actualXStart, actualXStop, actualYStart, actualYStop)
        bufferImage.squeezeImage()
        (refXStart, refXStop, refYStart, refYStop) = bufferImage.squeezedBox
        
        #sum reference and actual cooridates to shift squeezed box back into absolute location values
        squeezedBoxes.append((actualXStart+refXStart, actualYStart+refYStart, actualXStart+refXStop, actualYStart+refYStop))
        
    #cycle through boxes, pop boxes from list when they are found to be duplicates of the reference one.
    i = 0
    while i < endPoint:
        #for all other elements, left to right, calculate IOU
        copies = [j for j in range(i+1,endPoint+1) if Imaging.IOU(squeezedBoxes[i], squeezedBoxes[j]) > IOU_Allowance]

        #remove all copies from list
        squeezedBoxes = [val for index,val in enumerate(squeezedBoxes) if index not in copies]

        #update item in list to use next
        endPoint -= len(copies)
        i += 1      
    return squeezedBoxes

def FindRegionsOfInterest(binary, minPixels=900, maxPixels=100000):
    """
       Regions of interest are found based on found regions and elimination by
       intersection over union.
        
       @param binary     - CustomImage object - image object which will referenced to find regions for proposal.
       @param minPixels  - int                - sets min number of pixels required to be considered an official ROI.
       @param maxPixels  - int                - sets max number of pixels required to be considered an official ROI.
       @return success   - bool               - indicates if movement regions were able to be found.
       @return ROI_boxes - list of lists      - list of all ROI of the form "[x0, y0, x1, y1]".
    """
    #GET MOVEMENT REGIONS:
    success, MOVEMENT_boxes = GetMovementRegions(binary)
    if success is False:
        return False, []
        
    #GET REGION PROPOSALS FROM MOVEMENT REGIONS:
    #---intialize
    noRegionProposalsFound = True
    REGIONPROPOSAL_boxes = []
    
    #---scan
    for box in MOVEMENT_boxes:
        area = (box[2]-box[0])*(box[3]-box[1])
        if area > 100:
            CAND_box = GetRegionProposal(binary, box)
            area = (CAND_box[2]-CAND_box[0])*(CAND_box[3]-CAND_box[1])
            if minPixels < area and area < maxPixels:
                noRegionProposalsFound = False
                REGIONPROPOSAL_boxes.append(CAND_box)  
    
    #---assess   
    if noRegionProposalsFound:
        return False, []                
    
    #GET ROI FROM REGION PROPOSALS:
    ROI_boxes = GetRegionsOfInterest(binary, REGIONPROPOSAL_boxes)
    return success, ROI_boxes


#EOF