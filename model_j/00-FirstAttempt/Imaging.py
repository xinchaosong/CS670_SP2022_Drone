#!/usr/bin/env python

#Use: Library to contain all common tools used for interacting with images.

#Set Up Modules:
#--------------------------------------------------------------------------------------
#common
import numpy as np                              #library for working with arrays
# import matplotlib.pyplot as plt                 #libary for plotting (extension of numpy)
import re as regex                              #library for regular expressions
import cv2                                      #libary to solve computer vision problems
import math                                     #math tools
from os import listdir, makedirs                          #to use "listdir"
from os.path import isfile, join, exists        #to use file tools
# import random                                   #library for randomization tools
# import time                                     #library for timing tools
from image_tools.sizes import resize_and_crop   #ability to crop from middle
import copy

#mine
from CustomAssertions import CheckVariableIs    #libary with assertions and catch statements


#CLASSES:
#--------------------------------------------------------------------------------------    
#------------------------------------
#FrameManager------------------------
#------------------------------------
class FrameManager:
    """
        FrameManager class is responsible for finding images, keeping them on a stack, popping
        them when needed, and keeping track of paths.
    """
    def __init__(self, yz, x, rootFolder, dataSubFolder, resultsSubFolder, fileRegEx="[0-9_]+[.]jpg"):
        self.rootFolder = rootFolder
        self.dataSubFolder = dataSubFolder
        self.resultsSubFolder = resultsSubFolder
        self.subPath = r"%s\%s" % (yz,x)
        if yz == "":
            self.fullPathToSource = r"%s\%s" % (self.rootFolder, self.dataSubFolder)
            self.fullPathToResults = r"%s\%s" % (self.rootFolder, self.resultsSubFolder)
            self.imageSetName = "dataset"
        else:
            self.fullPathToSource = r"%s\%s\%s" % (self.rootFolder, self.dataSubFolder, self.subPath)
            self.fullPathToResults = r"%s\%s\%s" % (self.rootFolder, self.resultsSubFolder, self.subPath)
            self.imageSetName = "%s%s" % (yz,x)
        self.imageNames = [f for f in listdir(self.fullPathToSource) if isfile(join(self.fullPathToSource, f))]
        self.imageNames = ' '.join(self.imageNames)
        self.imageNames = regex.findall(fileRegEx,self.imageNames)
        self.imageNames.reverse()
        self.imageSetCount = 1
        self.imagesOnStack = max(len(self.imageNames), 0)
        self.currentImage = None
        self.currentImageName = None
        
        print(r"Number of images found --> ", self.imagesOnStack)
        if self.imagesOnStack > 0:
            self.__makeResultsFolder__()
            self._managerStatus = 0
        else:
            self._managerStatus = -1
            
    def __makeResultsFolder__(self):
        if not exists(self.fullPathToResults):
            makedirs(self.fullPathToResults)  
        
        
    #--------
    #GETTERS
    #--------
    def getCurrentFrame(self):
        return copy.deepcopy(self.currentImage)
        
    def getCurrentFrameResultsPath(self):
        return join(self.fullPathToResults,self.currentImageResultsName)
    
    def getRoiBoxes(self):
        return self.currentImageRoi_boxes
         
    def getRoi(self):
        return self.currentImageRoi
    
    
    #--------
    #SETTERS
    #--------
    def setNextFrame(self):
        if self.imagesOnStack > 0:
            self.currentImageName = self.imageNames.pop()
            self.currentImageResultsName = self.imageSetName + "_%d.jpg" % (self.imageSetCount)
            self.currentImage = CustomImage(self.currentImageName, join(self.fullPathToSource,self.currentImageName), "RGB")
            self.imagesOnStack -= 1
            self.imageSetCount += 1
            self.currentImageRoi = None
            self.currentImageRoi_boxes = None
        else:
            print("FrameManager: setNextFrame: There are no more images.")    
    
    def setCurrentFrame(self, customImageObject):
        self.currentImage = customImageObject
        
    def setCurrentImage(self, image):
        self.currentImage.image = image
        self.currentImage.shape = np.shape(image)
        
    def setRoiBoxes(self, boxes):
        self.currentImageRoi_boxes = boxes
        self.setRoi()
        
    def setRoi(self):
        self.currentImageRoi = []
        for box in self.currentImageRoi_boxes:
            buff = self.getCurrentFrame()
            buff.cropImage(box[0], box[2], box[1], box[3])
            self.currentImageRoi.append(buff.image) 
           
        
    #---------       
    #PRINTERS
    #---------
    def writeImageToResultsFolder(self):
        writeFlag = 0
        while writeFlag == 0:
            writeFlag = cv2.imwrite(self.getCurrentFrameResultsPath(),self.currentImage.image)



#------------------------------------
#Background--------------------------
#------------------------------------            
class Background:
    """
        Background class is a class to handle background subtraction and model updates.
    """
    def __init__(self, init_img, history=1, varThreshold=350, detectShadows=False):
        self.model = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
        self.update(init_img)
        self.mask = init_img
   
    def update(self, mask, changeModel=0):
        """
            Update background model and perform background subtraction.
            
            @param mask        - CustomImage object - image that will be background subtracted to yield a binary.
            @param changeModel - int                - (0 or 1), 1 indicates model is updated after subtraction.
        """
        self.mask = mask
        self.mask.image = self.model.apply(self.mask.image, None, changeModel)
        self.mask.blurImage(3)
        self.mask.quickBinarize() 
        
        #update properties of image
        self.mask.shape = np.shape(self.mask.image)


        
#------------------------------------
#CustomImage-------------------------
#------------------------------------   
class CustomImage:
    """
        CustomImage class is a class to handle importing, printing, altering and viewing images.
    """
    def __init__(self, imageName, imagePath, color):
        self.name = imageName
        self.import_color = color
        self.imagePath = imagePath
        self.__readImageFromPath__()
        self.rawImage = copy.deepcopy(self.image)
        self.shape = np.shape(self.image)
        
    def __readImageFromPath__(self):
        """  
            Read image from path, put image pixel values into into variable in cv2 style.
        """
        #check that path exists
        if exists(self.imagePath):
            if self.import_color.lower() == "RGB":
                self.image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
            elif self.import_color.lower() == "grayscale": 
                self.image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)
            elif self.import_color.lower() == "binary":
                self.image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)
                self.quickBinarize()
            else:
                self.image = cv2.imread(self.imagePath, cv2.IMREAD_UNCHANGED)
        else:
            raise FileNotFoundError("The file: \"" + self.imagePath + "\" does not exist")
        
    def __waitForUser__(self, time=0):
        cv2.waitKey(time)
        cv2.destroyAllWindows()         
        
        
    #--------     
    #GETTERS
    #--------
    def showImage(self):
        """
            Paints uint8 image to screen, and will collapse upon any user input signal.
        """
        cv2.imshow("Image",self.image)
        self.__waitForUser__()
        
    def showVideo(self):
        """
            Paints uint8 image to screen, and will collapse upon any user input signal.
        """
        cv2.imshow(self.name,self.image)
        self.__waitForUser__(100)       
        
    def showDistanceTransform(self):
        """
            Paints uint8 image of noramalized distance transform to screen, and will collapse upon any user input signal.
        """
        d = self.distTransform.copy()
        cv2.normalize(d, d, 0, 1.0, cv2.NORM_MINMAX)
        cv2.imshow(self.name,d)
        self.__waitForUser__()
        
    def getThresholdedDistanceTransform(self, allowance=1.0):
        """
            Returns all points in distance transform that are >= "allowance"*"max value in transform". 
            
            @param allowance - float         - fraction of maximum to set distance transform threshold.
            @return          - list of lists - all points in distance transform >= "allowance"*"max value in transform".
        """
        index = np.where(self.distTransform >= allowance*self.max_dT)
        return [[index[1][i],index[0][i]] for i in range(len(index[0]))]        
        
    def getBlackPoints(self):
        """
            Finds locations of all black points inside the "circle_radius" about
            the "circle_center" within the binary "image".
        """
        y, x = np.where(self.image == 0)
        return list(zip(x,y))
    
    def getWhitePoints(self):
        """
            Finds locations of all black points inside the "circle_radius" about
            the "circle_center" within the binary "image".
        """
        y, x = np.where(self.image == 255)
        return list(zip(x,y))    
    
    
    #--------   
    #SETTERS
    #--------
    def squeezeImage(self):
        """
            Finds smallest bounding box to encapsulate all white pixels of the main object.
            Set a value for "squeezedImage".
        """
        #calculate projections
        X_Projection = np.sum(self.image,axis=0)
        Y_Projection = np.sum(self.image,axis=1)

        #get largest white area in x direction
        X_Start, X_Stop = GetRangeOfWhitePixels(X_Projection)

        #get largest white area in y direction
        Y_Start, Y_Stop = GetRangeOfWhitePixels(Y_Projection)
        
        #update properties
        self.squeezedImage = self.image[Y_Start:Y_Stop,X_Start:X_Stop]
        self.squeezedBox = [X_Start,X_Stop,Y_Start,Y_Stop]
    
    def resizeImage(self, desired_size):
        """
            Scales image to a desired size. Aspect ratio maintained.
            
            @param desired_size - tuple - form (width,height) for desired size
        """
        assert(isinstance(desired_size,tuple))
        assert(len(desired_size)==2)
        
        #determine scale (keep aspect ratio)
        (Y_image, X_image) = self.shape[0], self.shape[1]
        if X_image > Y_image:
            width = int(desired_size[0])
            height = math.ceil((width/X_image)*Y_image)
        else:
            height = int(desired_size[1])
            width = math.ceil((height/Y_image)*X_image)
            
        #update properties
        self.image = cv2.resize(self.image, (width, height), interpolation=cv2.INTER_AREA)
        self.quickBinarize(1)
        self.shape = np.shape(self.image)
        
    def padImage(self, desired_size):
        """
            Pads an image with a constant color (black) to a desired size.
            
            @param desired_size - tuple - form (width, height) for desired size
        """
        assert(isinstance(desired_size,tuple))
        assert(len(desired_size)==2)
        
        #find amount of X and Y padding needed
        X_Pad_Needed = max(0,desired_size[0] - self.shape[1])
        Y_Pad_Needed = max(0,desired_size[1] - self.shape[0])

        #define even split of padding
        left = right = X_Pad_Needed//2
        top = bottom = Y_Pad_Needed//2       

        #if needed padding is odd, use floor and ceil
        if X_Pad_Needed % 2 > 0 and X_Pad_Needed > 0:
            left = int((X_Pad_Needed-1)/2)
            right = int((X_Pad_Needed+1)/2)
        if Y_Pad_Needed % 2 > 0 and Y_Pad_Needed > 0:
            top = int((Y_Pad_Needed-1)/2)
            bottom = int((Y_Pad_Needed+1)/2)
            
        #update properties
        self.image = cv2.copyMakeBorder(self.image, top, bottom, left, right, cv2.BORDER_CONSTANT) 
        self.shape = np.shape(self.image)
        
    def blurImage(self, size):
        """
            Performs Gaussian Blur as defined by cv2 module on image. 
            
            @param size - int - size of Gaussian filter to use (odd number between 1 and 11).
        """
        assert(isinstance(size, int))
        #update properties
        self.image = cv2.GaussianBlur(self.image,(size,size),0) 
        
    def cropImage(self, xStart, xStop, yStart, yStop):
        """
            Performs crop of image using coordinates.
            
            @param size - int - size of Gaussian filter to use (odd number between 1 and 11).
        """
        #update properties    
        self.image = self.image[yStart:yStop,xStart:xStop]
        self.shape = np.shape(self.image)
        
    def quickBinarize(self, threshold=200):
        """
            Simple binarize image using threshold value.
            
            @param threshold - int - value for thresholding (all numbers above thresh are forced to white).
        """
        assert(isinstance(threshold, int))
        #update properties
        self.image = cv2.threshold(self.image.copy(),threshold,255,cv2.THRESH_BINARY)[1]
        
        
    #--------- 
    #PRINTERS
    #---------
    def drawShapesOnImage(self, handAttrObj, predictionLabels):
        """
            Draw pertinant information to rawImage. Print this to file.
            
            @param handAttrObj      - HandAttributesManager obj - holds all hand measurements for 1 image.
            @param predictionLabels - list of ints              - holds gesture prediction labels for 1 image.
        """
        boxes                       = handAttrObj.getFinalBoundingBoxes()
        Crefs, Crefs_boundaryPoints = handAttrObj.getReferencePoints()
        COP_C, COP_R, COP_ER        = handAttrObj.getBubble()
        wristPts                    = handAttrObj.getWristPoints()
        
        for j in range(len(boxes)):
            #---box
            start_point = (boxes[j][0],boxes[j][1])
            end_point = (boxes[j][2],boxes[j][3])
            cv2.rectangle(self.rawImage,start_point,end_point,(150,0,0),1)       

            #---reference point and bounding points
            cv2.circle(self.rawImage, tuple(Crefs[j]), 5, 150, 1)   
            cv2.circle(self.rawImage, tuple(Crefs_boundaryPoints[j][0]), 5, 150, 1)   
            cv2.circle(self.rawImage, tuple(Crefs_boundaryPoints[j][1]), 5, 150, 1)   

            #---center of palm and expanded bubble
            self.rawImage[COP_C[j][1], COP_C[j][0]] = 0
            cv2.circle(self.rawImage, tuple(COP_C[j]), int(COP_R[j]), 150, 1)
            cv2.circle(self.rawImage, tuple(COP_C[j]), int(COP_ER[j]), 150, 1)

            #---wrist points
            cv2.circle(self.rawImage, tuple(wristPts[j][0]), 6, 150, 1)
            cv2.circle(self.rawImage, tuple(wristPts[j][1]), 6, 150, 1)

        #---gesture predictions
        cv2.putText(self.rawImage, "[%d]" %(predictionLabels[j]), end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
   
        #---replace image with raw image
        self.image = self.rawImage
    
    def printPixelValues(self, direction="all", setOrAll="all"):
        """
            Print pixel values of image row by row.
            
            @direction - string - print by column, row, or grid - options: "column", "row", "grid"
            @setOrAll  - string - print all values, or just unique values - options: "set", "all"
        """
        #make sure proper user inputs
        funcName = "printPixelValues"
        CheckVariableIs(funcName,"direction").String(direction)
        CheckVariableIs(funcName,"setOrAll").String(setOrAll)
        
        #quick access dictionary
        access = {"columnall":1, "columnset":2, "rowall":3, "rowset":4, "allall":5, "allset":6}
        combinedArgs = direction.lower() + setOrAll.lower()
        assert(combinedArgs in access.keys())
        option = access[combinedArgs]

        #print values
        listOfPixels = []
        if   option == 1:
            listOfPixels = [list(self.image[:,col]) for col in range(self.shape[1])]
        elif option == 2:
            listOfPixels = [set(self.image[:,col]) for col in range(self.shape[1])]
        elif option == 3:
            listOfPixels = [list(row) for row in self.image]
        elif option == 4:
            listOfPixels = [set(row) for row in self.image]
        elif option == 5:
            listOfPixels = [list(row) for row in self.image]
        elif option == 6:
            buffer = [list(set(row)) for row in self.image]
            listOfPixels = set(np.concatenate(buffer))
        else:
            raise Exception("Direction: " + direction.upper() + " is not a valid option.")            
        print(listOfPixels)
                
    def __str__(self):
        attributes = "Image name: " + self.name + "\nImport type: " + self.import_color + "\nImage path: " + self.imagePath
        return attributes


    
    

#--------------------------------------------------------------------------------------    
#PACKAGE-STATIC FUNCTIONS:
#--------------------------------------------------------------------------------------    
def showImage(image):
    """
        Paints a uint8 image to screen, and will collapse upon key mouse signal.
    """
    cv2.imshow("test image",image)
    key = cv2.waitKey(0) & 0xFF
    #cv2.destroyAllWindows()
      
def GetRangeOfWhitePixels(projection):
    """
        Find start and stopping points of range of white pixels in one dimension.
        
        @param projection - list of ints - pixel values of a single row/column of image (must be binary).
        @return           - list of ints - list of form [start, stop] of range of largest contiguous white pixels.
    """
    #get largest white area in range direction
    start = 0
    history = []
    for end in range(len(projection)):
        if projection[start] == 0 or projection[end] == 0:
            history.append([end-start,start,end])
            start = end
    history.append([end-start,start,end])
    history.sort(reverse=True,key=(lambda x: x[0]))
    return history[0][1:3]
    
def GetFULLRangeOfWhitePixels(projection):
    """
        Find start and stopping points of all white pixels in one dimension.
        
        @param projection - list of ints - pixel values of a single row/column of image (must be binary).
        @return           - list of ints - list of form [start, stop] of entire breadth of white pixels.
    """
    locations = np.where(np.array(projection) == 255)[0]
    if locations.size == 0:
        [start,stop] = [-1, -1]
    else:
        [start,stop] = [locations[0],locations[-1]]
    return [start,stop]
    
def squeezeImage(image):
    """
        Finds smallest bounding box to encapsulate all white pixels of the main object.
    """
    #calculate projections
    X_Projection = np.sum(image,axis=0)
    Y_Projection = np.sum(image,axis=1)

    #get largest white area in x direction
    X_Start, X_Stop = GetRangeOfWhitePixels(X_Projection)
        
    #get largest white area in y direction
    Y_Start, Y_Stop = GetRangeOfWhitePixels(Y_Projection)
    return image[Y_Start:Y_Stop,X_Start:X_Stop], [X_Start,X_Stop,Y_Start,Y_Stop]
    
def resizeImage(image, desired_size):
    """
        Scales image to a desired size. Aspect ratio maintained.

        @param desired_size - tuple - form (width,height) for desired size
    """
    assert(isinstance(desired_size,tuple))
    assert(len(desired_size)==2)
    image = cv2.resize(image, (desired_size[0], desired_size[1]), interpolation=cv2.INTER_AREA)
    return quickBinarize(image, 1)

def padImage(image, desired_size):
    """
        Pads an image with a constant color (black) to a desired size.

        @param desired_size - tuple - form (width, height) for desired size
    """
    assert(isinstance(desired_size,tuple))
    assert(len(desired_size)==2)

    #find amount of X and Y padding needed
    X_Pad_Needed = max(0,desired_size[0] - np.shape(image)[1])
    Y_Pad_Needed = max(0,desired_size[1] - np.shape(image)[0])

    #define even split of padding
    left = right = X_Pad_Needed//2
    top = bottom = Y_Pad_Needed//2       

    #if needed padding is odd, use floor and ceil
    if X_Pad_Needed % 2 > 0 and X_Pad_Needed > 0:
        left = int((X_Pad_Needed-1)/2)
        right = int((X_Pad_Needed+1)/2)
    if Y_Pad_Needed % 2 > 0 and Y_Pad_Needed > 0:
        top = int((Y_Pad_Needed-1)/2)
        bottom = int((Y_Pad_Needed+1)/2)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  
    
def obtainThresholdValues(image, clusterBoxes=[]):
    """
        Finds appropriate binarization value for an image to separate the hand regions from the background
        and other objects by using the pixel values in the cluster regions. The cluster regions are expanded
        slightly so as to hopefully include some of the background pixel values so that Otzu's method can be
        used to identify the most appropriate binarization value by observing the background peak pixel value
        and the hand peak pixel value. The output value threshold is an array of threshold values, one for 
        each clusterBox.
    """
    #initialize
    expansionMargin = 0
    
    #blur image to reduce temperature gradients
    image = cv2.GaussianBlur(image,(3,3),0)
    
    #obtain threshold value
    if len(clusterBoxes) == 0:
        threshold = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    else:
        threshold = []
        for i in range(len(clusterBoxes)):
            #obtain expanded box
            x_start = max(0,clusterBoxes[i][0]-expansionMargin)
            y_start = max(0,clusterBoxes[i][1]-expansionMargin)
            x_end = min(image.shape[1],clusterBoxes[i][2]+expansionMargin)
            y_end = min(image.shape[0],clusterBoxes[i][3]+expansionMargin)
            
            #perform binarization on original image in bounding rectangle
            binarizeArea = image[y_start:y_end,x_start:x_end]
            threshold.append(int(cv2.threshold(binarizeArea,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]))
    return threshold
    
def quickBinarize(image, threshold):
    """
        Performs a simple binarization of an image using a given threshold.
        Returns the binarized image.
    """
    return cv2.threshold(image.copy(),int(threshold),255,cv2.THRESH_BINARY)[1]

def findLargestContour(contours):
    """Find largest contour of a set of contours"""
    if not contours:
        return -1
    else:
        return max(contours,key=cv2.contourArea)
    
def ContoursToPoints(contours):
    """
        Changes format of contour points to a list of tuples. Contours use the top left as the image origin
        but this will return points corresponding to a bottom left origin.
    """
    Point_Array = []
    for i in range(len(contours)):
        Point_Array.append((contours[i][0]).tolist())
    return Point_Array

def DefectsToPoints(contourArray, defectArray):
    points = []
    for i in range(defectArray.shape[0]):
        s, e, f, d = defectArray[i][0]
        points.append(list(contourArray[f][0]))
    return points

def GetBoxCenters(boxes):
    """
       Finds center coordinates for each box in a list of boxes.
       "boxes" should be of the form (x0,y0,xf,yf); 0 is the 
       top left corner of an image, and f is the bottom right
       corner of an image.
    """
    centers = []
    for i in range(len(boxes)):
        x_width = (boxes[i][2] - boxes[i][0])
        y_width = (boxes[i][3] - boxes[i][1])
        x_mid = x_width/2 + boxes[i][0]
        y_mid = y_width/2 + boxes[i][1]
        centers.append([x_mid,y_mid]) 
    return centers

def StretchBox(box, image, stretchSize):
    """
        Returns a box that has been expanded by "advancement" number of pixels.
        Box will not expand past dimensions of image. If "stretchSize" is given
        as a value between 0 and 1, it is taken to be a percentage of the size
        of the image in each dimensions; if "strechSize" is given as any other 
        value greater than 1, the stretch will be discreet (simple addition).
    """
    #determine stretch size (by percentage or discreetly)
    if stretchSize < 1:
        stretchSizeX = int(stretchSize*image.shape[0])
        stretchSizeY = int(stretchSize*image.shape[1])
    else:
        stretchSizeX = int(stretchSize)
        stretchSizeY = int(stretchSize)
    
    #perform stretch
    xstart = max(0, box[0]-stretchSizeX)
    ystart = max(0, box[1]-stretchSizeY)
    xstop = min(box[2]+stretchSizeX, image.shape[1])
    ystop = min(box[3]+stretchSizeY,image.shape[0])
    return [xstart, ystart, xstop, ystop]


def IOU(a, b, epsilon=1e-5):
    """
        Intersection over union to identify the most unique region proposals so that there are not duplicate images.
    """
    #coordinates of intersection box
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    #area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    #combined area
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap
    return area_overlap / (area_combined+epsilon)


#EOF