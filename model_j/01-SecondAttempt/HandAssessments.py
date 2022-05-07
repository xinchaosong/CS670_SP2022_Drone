#!/usr/bin/env python

#Use: Library to contain all common tools used for interacting with images.

#Set Up Modules:
#--------------------------------------------------------------------------------------
#common modules
import cv2                         #libary to solve computer vision problems
import time                        #library for timing tools
import copy
from math import floor
import numpy as np
import random
from itertools import compress     #used to allow boolean to choose elements of list

#my modules
import Distance
import Imaging


#--------------------------------------------------------------------------------------
#Main Algorithm
#--------------------------------------------------------------------------------------
class HandAttributesManager:
    def __init__(self):
        self.bubbleCenter = []
        self.bubbleRadius = []
        self.extendedBubbleRadius = []
        self.wristPoints = []
        self.finalHandImages = []
        self.finalBoundingBoxes = []
        self.referencePoint = []
        self.referencePointBoundaries = []

    #---------------------------------------------------------------------------
    #GETTERS
    #---------------------------------------------------------------------------   
    def getBubble(self):
        return [self.bubbleCenter, self.bubbleRadius, self.extendedBubbleRadius]
        
        
    def getWristPoints(self):
        return self.wristPoints
    
    
    def getFinalHandImages(self):
        return self.finalHandImages        
 

    def getFinalBoundingBoxes(self):
        return self.finalBoundingBoxes

        
    def getReferencePoints(self):
        return [self.referencePoint, self.referencePointBoundaries]
        
        
    #---------------------------------------------------------------------------
    #SETTERS
    #---------------------------------------------------------------------------   
    def addBubbles(self, COP_C, COP_R, COP_ER):
        self.bubbleCenter.append(COP_C)
        self.bubbleRadius.append(COP_R)
        self.extendedBubbleRadius.append(COP_ER)    
        
        
    def addWristPoints(self, pointSet):
        self.wristPoints.append(pointSet)    
        
        
    def addFinalHandImage(self, img):
        self.finalHandImages.append(img)
        
        
    def addFinalBoundingBox(self, box):
        self.finalBoundingBoxes.append(box)
        
        
    def addReferencePoints(self, Cref, Cref_boundaryPoints):
        self.referencePoint.append(Cref)
        self.referencePointBoundaries.append(Cref_boundaryPoints)

        
    #---------------------------------------------------------------------------
    #PRINTERS
    #---------------------------------------------------------------------------   
    def __str__(self):
        buffer = []
        buffer = "Bubble Centers: \n"
        for bubble in self.bubbleCenter:
            buffer += str(bubble) + "\n"
            
        buffer += "Bubble Radii: \n"
        for radius in self.bubbleRadius:
            buffer += str(radius) + "\n"
    
        buffer += "Extended Bubble Raddi: \n"
        for radius in self.extendedBubbleRadius:
            buffer += str(radius) + "\n"    
    
        buffer += "Wrist Points: \n"
        for pointSet in self.wristPoints:
            buffer += str(pointSet) + "\n"     
            
        buffer += "Reference Points: \n"
        for pointSet in self.referencePoint:
            buffer += str(pointSet) + "\n" 
        
        buffer += "Reference Boundary Points: \n"
        for pointSet in self.referencePointBoundaries:
            buffer += str(pointSet) + "\n"     
        
        return buffer
    
    

#-------------------------------------------------------------------------------------        
#HELPER FUNCTIONS FOR BUBBLE GROWTH:    
#-------------------------------------------------------------------------------------     
def getCestCandidates(binary, ROI_box, allowance=0.95):
    """
        Find potential COP estimated bubble center values based on distance transform.
        
        @param binary    - CustomeImage object - image for which potential Cest candidates will be found.
        @param regionBox - list of lists       - list of all ROI of the form "[x0, y0, x1, y1]".
        @param allowance - float               - fraction of maximum to set distance transform threshold.
        @return          - list of lists       - list of all points [x,y] for good Cest candidates.
    """
    binary.cropImage(ROI_box[0], ROI_box[2], ROI_box[1], ROI_box[3])
    centerChoices = binary.getThresholdedDistanceTransform(allowance)
    for i in range(len(centerChoices)):
        centerChoices[i][1] += ROI_box[1] #use reference point to get actual location relative to image
        centerChoices[i][0] += ROI_box[0] #use reference point to get actual location relative to image
    return centerChoices


def getCref(binary, ROI_box, reference):
    """
        Get arm penetration reference point. This point indicates the edge belonging to the wrist.
        
        NOTE: if using reference model (FindArmPenetrationReferencePoint2), then the following applies:
              [0=DOWN; 1=RIGHT; 2=UP; 3=LEFT] <--- reference values
        
        
        @param binary    - CustomeImage object - image for which potential Cest candidates will be found.
        @param regionBox - list of lists       - list of all ROI of the form "[x0, y0, x1, y1]".
        @return [0]      - list                - singular point [x,y] for Cref.
        @return [1]      - list of lists       - list of points [x,y] that bound Cref on one dimension.
    """
#     direction = {-1:"manual", 0:"DOWN", 1:"RIGHT", 2:"UP", 3:"LEFT"}
#     print("Value of reference used is:", direction[reference])
    return findCestAndCestBoundaryPoints(binary.image.copy(), ROI_box, reference)


def getCest(cestCandidates, Cref):
    """
        Finds which Cest is furthest from Cref. This is chosen as Bubble Growth's Cest.
        
        @param cestCandidates - list of lists - list of all points [x,y] for good Cest candidates.
        @param Cref           - list          - singular point [x,y] for Cref.
        @return Cest          - list          - singular point [x,y] for Cest.
    """
    return Distance.FarthestPoint(cestCandidates, Cref)
    

    
def findCestAndCestBoundaryPoints(image, regionBox, reference=-1):
    """
        Finds which edge of the "regionBox" contains the most white pixels. This is assumed to be the side of
        the boundary that contains the arm. With this assumption, the midpoint of the edge found to have 
        the most white pixels is found and returned. This point will be the reference point when determining
        which distance transform point is found.
        [-1=INSPECT_ALL_EDGES; 0=DOWN; 1=RIGHT; 2=UP; 3=LEFT]
        
        @param image     - cv2 object - image being inspected for reference edge.
        @param regionBox - list       - bounding box that defines 1 ROI for image, in form [x0,y0,x1,y1].
        @param reference - signed int - indication to use manual (all edge inspection) or just one edge.
        @return - 
    """
    #(OPTION: -1) PERFORM MANUAL FINDING OF REFERENCE EDGE
    if reference == -1:
        #count number of pixels on border of image
        yLeftStart,yLeftStop   = Imaging.GetRangeOfWhitePixels(image[regionBox[1]:regionBox[3],regionBox[0]])
        yRightStart,yRightStop = Imaging.GetRangeOfWhitePixels(image[regionBox[1]:regionBox[3],regionBox[2]])
        xLeftStart,xLeftStop   = Imaging.GetRangeOfWhitePixels(image[regionBox[1],regionBox[0]:regionBox[2]])
        xRightStart,xRightStop = Imaging.GetRangeOfWhitePixels(image[regionBox[3],regionBox[0]:regionBox[2]])

        #find total number of non-black pixels are on each edge
        array = [np.count_nonzero(image[regionBox[1],regionBox[0]:regionBox[2]]), #going down
                 np.count_nonzero(image[regionBox[1]:regionBox[3],regionBox[2]]), #going right
                 np.count_nonzero(image[regionBox[3],regionBox[0]:regionBox[2]]), #going up
                 np.count_nonzero(image[regionBox[1]:regionBox[3],regionBox[0]])] #going left    
    
        #find a centerpoint of edge where arm is penetrating, error if no white pixels on border
        indexOfMax = array.index(max(array))
        if indexOfMax == 0:
            referencePoint = [regionBox[0] + xLeftStart + int((xLeftStop-xLeftStart)/2), regionBox[1]]
            boundaryPoints = [[regionBox[0] + xLeftStart, regionBox[1]], [regionBox[0] + xLeftStop, regionBox[1]]]
        elif indexOfMax == 1:
            referencePoint = [regionBox[2], regionBox[1] + yRightStart + int((yRightStop-yRightStart)/2)]
            boundaryPoints = [[regionBox[2], regionBox[1] + yRightStart], [regionBox[2], regionBox[1] + yRightStop]]
        elif indexOfMax == 2:
            referencePoint = [regionBox[0] + xRightStart + int((xRightStop-xRightStart)/2), regionBox[3]]
            boundaryPoints = [[regionBox[0] + xRightStart, regionBox[3]], [regionBox[0] + xRightStop, regionBox[3]]]        
        else:
            referencePoint = [regionBox[0], regionBox[1] + yLeftStart + int((yLeftStop-yLeftStart)/2)]
            boundaryPoints = [[regionBox[0], regionBox[1] + yLeftStart], [regionBox[0], regionBox[1] + yLeftStop]]
            
    #(OPTION: 0-3) SKIP RIGHT TO REFERENCE EDGE (INFORMED BY REFERENCE MODEL)
    elif reference == 0:
        start,stop = Imaging.GetRangeOfWhitePixels(image[regionBox[1],regionBox[0]:regionBox[2]]) #top side (going down)
        referencePoint = [regionBox[0] + start + int((stop-start)/2), regionBox[1]]
        boundaryPoints = [[regionBox[0] + start, regionBox[1]], [regionBox[0] + stop, regionBox[1]]]        
    elif reference == 1:
        start,stop = Imaging.GetRangeOfWhitePixels(image[regionBox[1]:regionBox[3],regionBox[2]]) #right side (going left)
        referencePoint = [regionBox[2], regionBox[1] + start + int((stop-start)/2)]
        boundaryPoints = [[regionBox[2], regionBox[1] + start], [regionBox[2], regionBox[1] + stop]]        
    elif reference == 2:
        start,stop = Imaging.GetRangeOfWhitePixels(image[regionBox[3],regionBox[0]:regionBox[2]]) #bottom side (going up)
        referencePoint = [regionBox[0] + start + int((stop-start)/2), regionBox[3]]
        boundaryPoints = [[regionBox[0] + start, regionBox[3]], [regionBox[0] + stop, regionBox[3]]]        
    elif reference == 3:
        start,stop = Imaging.GetRangeOfWhitePixels(image[regionBox[1]:regionBox[3],regionBox[0]]) #left side (going right)
        referencePoint = [regionBox[0], regionBox[1] + start + int((stop-start)/2)]
        boundaryPoints = [[regionBox[0], regionBox[1] + start], [regionBox[0], regionBox[1] + stop]]        
    else:
        raise Exception("findCestAndCestBoundaryPoints: That is not an appropriate reference.") 
    
    return referencePoint, boundaryPoints
     
    
def GenerateHandBorderPoints(roi, xOffset=0, yOffset=0):
    """
        Find points that border the hand based on the image's largest contour.
        
        @param roi            - cv2 image object - image of region of interest (already squeezed).
        @param xOffset        - int              - shift in x direction to make points absolute to original image. 
        @param yOffset        - int              - shift in y direction to make points absolute to original image. 
        @return All_Points    - list of lists    - list of [x,y] points in hand contour.
        @return Defect_Points - list of lists    - list of [x,y] defect points in hand contour.
    """
    contours = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    largestContour = Imaging.findLargestContour(contours)
    hullconvex = cv2.convexHull(largestContour, returnPoints = False)
    
    #GET ALL CONTOUR POINTS
    All_Points = Imaging.ContoursToPoints(largestContour)
    All_Points = [[point[0] + xOffset, point[1] + yOffset] for point in All_Points]    
    
    #TRY TO GET DEFECT POINTS (makes COP algorithm more robust)
    try:
        defects = cv2.convexityDefects(largestContour, hullconvex)
        Defect_Points = Imaging.DefectsToPoints(largestContour, defects)
        Defect_Points = [[point[0] + xOffset, point[1] + yOffset] for point in Defect_Points]  
    except:
        Defect_Points = []

    return All_Points, Defect_Points    
    
    
#-------------------------------------------------------------------------------------------------------
#--------------------------------------BUBBLE GROWTH METHOD---------------------------------------------
#-------------------------------------------------------------------------------------------------------
def BubbleGrowth(Cest, Cest_R, borderPoints):
    """
        Finds center of palm coordinate and corresponding radius. (Max inscribed circle).
        
        @param  Cest         - list         - singular point [x,y] for Cest (initial COP).
        @param  borderPoints - list of list - list of poitns [x,y] in hand contour.
        @return current_C    - list         - singular point [x,y] for final bubble center (COP).
        @return current_R    - float        - value for the radius of final bubble.
    """
    current_C = Cest
    current_R = Cest_R
    moving = True
    visited = []
    while (moving):
        moving = False
        for point in borderPoints:
            candidate_C = Distance.ShortAdvancement(current_C, point, 0.13)
            if candidate_C in visited:
                continue
            candidate_R = Distance.MinimumDistance(borderPoints, candidate_C)
            if candidate_R > current_R:
                current_C, current_R = candidate_C, candidate_R
                moving = True
                break
            visited.append(candidate_C)
    return current_C, current_R
#-------------------------------------------------------------------------------------------------------
#--------------------------------------BUBBLE GROWTH METHOD---------------------------------------------
#-------------------------------------------------------------------------------------------------------    




#-------------------------------------------------------------------------------------        
#HELPER FUNCTIONS FOR BUBBLE SEARCH:    
#-------------------------------------------------------------------------------------  
def RemoveArmByWristPoints(binary, wristPoints, COP_C):
    """
        Removes arm from a binary image at the wrist points, uses the "bubbleCenter" 
        as a means of determining which side of the wrist points is the arm.
        
        @param binary      - cv2 image object - simple image of full binary image.
        @param wristPoints - list of lists    - list of points [x,y] for two wrist points.
        @param COP_C       - list             - singular point [x,y] for final bubble center (COP).
        @return newBinary  - cv2 iamge object - same as "binary" but with everything but hand removed.
    """
    #unpack points
    pt1, pt2 = wristPoints
    
    #obtain inequality based on wrist points
    try:
        m = float(pt2[1] - pt1[1])/float(pt2[0] - pt1[0])
    except ZeroDivisionError:
        m = 0
    b = (-1*m*pt1[0] + pt1[1])
    
    #genearte new image location, don't write on top of image
    newBinary = binary.copy()
    
    #generate points for image (grid of values); separate x and y values for matrix operations
    y_groundTruth, x_groundTruth = np.where(newBinary >=0)
    x_groundTruth = x_groundTruth.tolist()
    y_groundTruth = y_groundTruth.tolist()
    
    #find which points to remove
    side1 = y_groundTruth >= np.array(m*np.array(x_groundTruth) + b)
    try:
        if side1[x_groundTruth.index(COP_C[0]) + y_groundTruth.index(COP_C[1])] == True:
            xToDelete = list(compress(x_groundTruth, ~side1))
            yToDelete = list(compress(y_groundTruth, ~side1))
        else:
            xToDelete = list(compress(x_groundTruth, side1))
            yToDelete = list(compress(y_groundTruth, side1))

        #cut off wrist
        for i in range(len(xToDelete)):
            newBinary[yToDelete[i],xToDelete[i]] = 0
    except:
        raise
    return newBinary


#-------------------------------------------------------------------------------------------------------
#--------------------------------------BUBBLE SEARCH METHOD---------------------------------------------
#-------------------------------------------------------------------------------------------------------  
def BubbleSearch(contourPts, COP_R, COP_C, Cref, Cref_boundaryPoints):
    """
        Finds wrist points for a hand binary based on contour points.
        
        @param contourPts            - list of list       - list of [x,y] points for hand contour.
        @param COP_R                 - float              - value for the radius of final bubble. 
        @param COP_C                 - list               - singular point [x,y] for final bubble center (COP).
        @param Cref                  - list               - singular point [x,y] for Cref.
        @param Cref_boundaryPoints   - list of lists      - list of points [x,y] that bound Cref on one dimension.
        @return [0]                  - list of lists      - list of points [x,y] for two wrist points.
        @return expandedBubbleRadius - float              - value for radius of exanded bubble used to find wrist pts. 
    """
    #initialization
    expandedBubbleRadius = 1.15*COP_R  #initial expanded bubble radius (USED TO BE 1.4)
    maxDist              = 1.87*COP_R  #minimum wrist width acceptable (USED TO BE 1.9)
    minDist              = 0.94*COP_R  #maximum wrist width acceptable (USED TO BE 1.1)
    meetsCriteria        = []         #list of points meeting all candidate criteria
    whileCounter         = 0          #prevents infinite while loop
    whileCounterLimit    = 10         #max iterations to prevent infinite while loop
    keepLooking          = True       #enter while loop
    
    #determine to search for wrist points, or accept the ones on border of image as the points
    maxDistanceFromCref  = Distance.EuclideanDistance([COP_C], Cref)
    if expandedBubbleRadius > maxDistanceFromCref:
        return Cref_boundaryPoints, expandedBubbleRadius

    #find wrist points
    #note: expand bubble, find points, check against criteria, repeat as necessary.
    while keepLooking:
        #limit scope of points to those inside expanded bubble
        ptsInBubble, _ = Distance.PointsRelativeToCircle(contourPts, COP_C, expandedBubbleRadius)
        ptsInBubble.append(ptsInBubble[0])
        
        #calculate distance between each contiguous set of points
        ptDist = []
        meetsCriteria = []
        for i in range(len(ptsInBubble) - 1):
            [p1, p2] = ptsInBubble[i:i+2]
            midPoint = Distance.CenterOfMass([p1,p2])
            ptDist.append([Distance.EuclideanDistance([p1], p2),Distance.EuclideanDistance([midPoint], Cref),p1,p2])
            if ptDist[i][0] > minDist and ptDist[i][0] < maxDist and ptDist[i][1] < maxDistanceFromCref:
                meetsCriteria.append(ptDist[i])     
        
        #obtain wrist points
        if len(meetsCriteria) >= 1:                             #if candidates present, take candidates closest to Cref
            keepLooking = False
            meetsCriteria.sort(reverse=False,key=(lambda x: x[1]))
            [wp1,wp2] = meetsCriteria[0][2:4]
        elif whileCounter == whileCounterLimit:                 #if no candidates present, take points bounding Cref
            keepLooking = False
            [wp1,wp2] = Cref_boundaryPoints[0:2]
        else:
            expandedBubbleRadius *= 1.07   #(USED TO BE 1.011)
        whileCounter += 1
    return [wp1,wp2], expandedBubbleRadius    
#-------------------------------------------------------------------------------------------------------
#--------------------------------------BUBBLE SEARCH METHOD---------------------------------------------
#-------------------------------------------------------------------------------------------------------     
    
    
def SqueezePadResize(binary, box, padSize=5, finalSize=(100,100)):
    """
        Squeeze image into smallest bounding box. Pad and resize to required size for 
        training set.
        
        @param binary    - cv2 image object - image of the already-squeezed hand.
        @param box       - list of int      - region of interest in form [x0,y0,x1,y1].
        @param padSize   - int              - used to set the padding width before resizing.
        @param finalSize - (int,int)        - used to set the size of final output image.
        @return resized  - cv2 image object - iamge of the final, final, final hand image.
    """
    #get region
    img = binary[box[1]:box[3],box[0]:box[2]]
    
    #squeeze
    img, newBox = Imaging.squeezeImage(img)
    
    #calculate new bounding box, because we squeezed
    finalBox = [newBox[0]+box[0], 
                newBox[2]+box[1], 
                newBox[1]+box[0], 
                newBox[3]+box[1]]
    
    
    #pad and resize
    maxY, maxX = np.shape(img)
    desired_s = max(maxX+padSize,maxY+padSize)
    padded = Imaging.padImage(img, desired_size=(desired_s,desired_s)) 
    resized = Imaging.resizeImage(padded, desired_size=finalSize)
    return resized, finalBox
    

    
    
#-------------------------------------------------------------------------------------        
#MAIN FUNCTIONS:    
#-------------------------------------------------------------------------------------    
def AssessHandFeatues(image, ROI, ROI_boxes, reference):
    """
        Perform the Bubble Growth and Bubble Search methods to find all attributes for
        a HandAttributesManager object.
        
        @param image           - CustomImage object - image that will be assessed
        @param ROI_boxes       - list of lists      - bounding boxes that define ROI for image, in form [x0,y0,x1,y1].
        @param reference       - list of ints       - [-1=manual Cref assessment; 0,1,2,3 uses Cref NN to find Cref].
        @return success        - bool               - True if everything runs smoothly
        @return handAttributes - HandAttributesManager object - contains values for all assessments from 1 execution.
        
    """
    #initialize
    handAttributes = HandAttributesManager()
    success = True

    #iterate over all boxes
    for box,roi,Cref_reference in list(zip(ROI_boxes,ROI,reference)):
        try:
            #BUBBLE GROWTH METHOD:
            #--------------------------------------------------------------------------------------
            #get Cref and Cest
            Cest_candidates = getCestCandidates(copy.deepcopy(image), box, allowance=0.80) #(allowance USED TO BE 0.80)
            Cref, Cref_boundaryPoints = getCref(copy.deepcopy(image), box, reference=Cref_reference)
            Cest = getCest(Cest_candidates, Cref)

            #get contour points of hand
            #---grab contour points (points are ABSOLUTE to original image)
            allPoints, defectPoints = GenerateHandBorderPoints(roi, xOffset=box[0], yOffset=box[1])

            #---reduce number of contour points
            n_pts_for_BG = max(40, len(allPoints)*0.16)    #use 16% of contour poitns, but no less than 40 
            scale = floor(len(allPoints)/n_pts_for_BG)
            if scale > 1:                                  #if there are >= 40 points, strip points
                optimalPoints = Distance.StripInterleavedPoints(allPoints, defectPoints, scale)
            else:
                optimalPoints = copy.deepcopy(allPoints)
            random.shuffle(optimalPoints) #randomly shuffle to increase speed of BG method.
                
            #initialize value for radius of Cest
            Cest_bubbleRadius = Distance.MinimumDistance(optimalPoints,Cest)

            #bubble growth algorithm
            COP_C, COP_R = BubbleGrowth(Cest, Cest_bubbleRadius, optimalPoints)

            #BUBBLE SEARCH METHOD:
            #--------------------------------------------------------------------------------------    
            #bubble search
#             wristPts, COP_ER = BubbleSearch(optimalPoints, COP_R, COP_C, Cref, Cref_boundaryPoints) #MISTAKE! I SHOULD USE ALL PTS!
            wristPts, COP_ER = BubbleSearch(allPoints, COP_R, COP_C, Cref, Cref_boundaryPoints)
            
            #HAND SEGMENTATION (ARM REMOVAL):
            #--------------------------------------------------------------------------------------  
            binaryWithoutArm = RemoveArmByWristPoints(image.image.copy(), wristPts, COP_C)

            #final image adjustments
            finalImage, finalBox = SqueezePadResize(binaryWithoutArm, box, padSize=5, finalSize=(100,100))  

            #ADD ALL THINGS TO HAND ATTRIBUTE MANAGER:
            #--------------------------------------------------------------------------------------  
            handAttributes.addBubbles(COP_C, COP_R, COP_ER)
            handAttributes.addReferencePoints(Cref, Cref_boundaryPoints)
            handAttributes.addWristPoints(wristPts)
            handAttributes.addFinalHandImage(finalImage)
            handAttributes.addFinalBoundingBox(finalBox)
        except Exception as e:
            raise
                
        if len(handAttributes.getFinalHandImages()) == 0:
            success = False    
    return success, handAttributes


#EOF