# Set Up Modules:
# --------------------------------------------------------------------------------------
import numpy as np  # library for working with arrays
import cv2  # libary to solve computer vision problems
import selectivesearch
import pickle
import copy
import os
from os.path import join
import keras
from keras.models import model_from_json
from keras.backend import manual_variable_initialization

manual_variable_initialization(True)
from PIL import Image


# --------------------------------------------------------------------------------------
class BoxDetector:
    def __init__(self, printBoxesToImages=False, preScaleFactor=0.30, SS_SigmaGaussFilter=1.0, SS_Scale=350,
                 predictionConfidenceThreshold=0.85, IOU_Allowance=0.40):
        # load box detection model
        self.box_detection_model = self.__load_model__()

        # obtain prediction options
        self.imageScaleFactor = preScaleFactor
        self.sigma = SS_SigmaGaussFilter
        self.k = SS_Scale
        self.predictionConfidence = predictionConfidenceThreshold
        self.IOU_Allowance = IOU_Allowance

        # data collected from running
        self.numberOfBoxesDetected = 0
        self.numberOfBoxesChosen = 0
        self.allBoxes = []
        self.allBoundingBoxes = []
        self.allBoxes_AfterNMS = []
        self.allBoxes_Final = []
        self.allBoundingBoxConfidences = []
        self.boundingBoxes = []

        # other options
        self.printBoxesToImages = printBoxesToImages
        self.currentImage = None

    def __load_model__(self):
        # ---load json string, unpack model into object
        json_file = open(join(os.getcwd(), 'MODEL_1_Apr4_2022.json'), 'r')
        try:
            loaded_model_json = json_file.read()
            loaded_model_object = model_from_json(loaded_model_json)
        except:
            json_file.close()
            raise Exception(
                "Could not find Model 'JSON' File. Make sure it's in same directory as Box_Object_Detector.py.")

        # ---load weights into model
        try:
            loaded_model_object.load_weights(join(os.getcwd(), 'MODEL_1_Apr4_2022.h5'))
        except:
            raise Exception(
                "Could not find Model 'H5' File. Make sure it's in same directory as Box_Object_Detector.py.")

        return loaded_model_object

    def __scale_image__(self):
        (y, x, z) = np.shape(self.currentImage)
        newSize = (int(self.imageScaleFactor * x), int(self.imageScaleFactor * y))
        self.currentImage = cv2.resize(self.currentImage, (newSize), interpolation=cv2.INTER_LANCZOS4)

    def __makeActualPrediction__(self):
        listOfBoxBoxes = []
        confidenceLevels = []
        for (x, y, w, h) in self.allBoxes:
            croppedImage = self.currentImage[y:y + h, x:x + w]
            grayScaleImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
            snippetSize = np.shape(grayScaleImage)
            if snippetSize[0] * snippetSize[1] > 0:
                regionProposal = cv2.resize(grayScaleImage, (100, 100), interpolation=cv2.INTER_LANCZOS4)
                regionProposal = regionProposal.astype('float32')
                regionProposal /= 255.0
                regionProposal = np.reshape(regionProposal, (-1, 100, 100, 1))
                prediction = self.box_detection_model(regionProposal, training=False).numpy()
                if prediction[0][1] > self.predictionConfidence:
                    listOfBoxBoxes.append([x, y, w, h])
                    confidenceLevels.append(prediction[0][1])
        self.allBoundingBoxes = listOfBoxBoxes
        self.allBoundingBoxConfidences = confidenceLevels

    def __NonMaximumSuppression__(self):
        boxConfidence = list(zip(self.allBoundingBoxes, self.allBoundingBoxConfidences))
        boxConfidence.sort(key=lambda x: x[1], reverse=True)

        # cycle through boxes, pop boxes from list when they are found to be duplicates of the reference one.
        endPoint = len(boxConfidence) - 1
        i = 0
        while i < endPoint:
            # for all other elements, left to right, calculate IOU
            copies = [j for j in range(i + 1, endPoint + 1) if
                      self.__IOU__(boxConfidence[i][0], boxConfidence[j][0]) > self.IOU_Allowance]

            # remove all copies from list
            boxConfidence = [val for index, val in enumerate(boxConfidence) if index not in copies]

            # update item in list to use next
            endPoint -= len(copies)
            i += 1

        self.allBoxes_AfterNMS = [item[0] for item in boxConfidence]

    def __IOU__(self, a, b, epsilon=1e-5):
        """
            Intersection over union to identify the most unique region proposals so that there are not duplicate images.
        """
        # coordinates of intersection box
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[0] + a[2], b[0] + b[2])
        y2 = min(a[1] + a[3], b[1] + b[3])

        # area where the boxes intersect
        area_overlap = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # combined area
        area_a = (a[2] + 1) * (a[3] + 1)
        area_b = (b[2] + 1) * (b[3] + 1)
        area_combined = area_a + area_b - area_overlap
        return abs(area_overlap / (area_combined + epsilon))

    def __eliminateInnerBoxes__(self):
        boxes = self.allBoxes_AfterNMS
        boxes.sort(key=lambda x: ((x[0] + x[2]) * (x[1] + x[3])), reverse=True)

        # cycle through boxes, pop boxes from list when they are found to be duplicates of the reference one.
        endPoint = len(boxes) - 1
        i = 0
        while i < endPoint:
            # for all other elements, left to right, calculate IOU
            copies = [j for j in range(i + 1, endPoint + 1) if self.__isBoxInsideBox__(boxes[j], boxes[i])]

            # remove all copies from list
            boxes = [val for index, val in enumerate(boxes) if index not in copies]

            # update item in list to use next
            endPoint -= len(copies)
            i += 1

        self.allBoxes_Final = boxes
        self.numberOfBoxesChosen = len(self.allBoxes_Final)

    def __isBoxInsideBox__(self, box1, box2):
        condition1 = box1[0] >= box2[0]
        condition2 = box1[1] >= box2[1]
        condition3 = box1[0] + box1[2] <= box2[0] + box2[2]
        condition4 = box1[1] + box1[3] <= box2[1] + box2[3]
        return condition1 and condition2 and condition3 and condition4

    def __printBoxesToImages__(self):
        for (x, y, w, h) in self.allBoxes_Final:
            cv2.rectangle(self.originalImage, (x, y), (x + w, y + h), [255, 105, 180], 2)

    def __printBoxesToSmallerImages__(self):
        for (x, y, w, h) in self.allBoxes_Final:
            cv2.rectangle(self.currentImage, (x, y), (x + w, y + h), [255, 105, 180], 2)

    def __expand_bounding_box__(self, BBox):
        # grab points
        x1, y1, x2, y2 = BBox[0], BBox[1], BBox[0] + BBox[2], BBox[1] + BBox[3]
        print("These are the points: ", x1, y1, x2, y2)
        # scaling information
        enlarge = (1 / self.imageScaleFactor)
        eta = enlarge

        # Image Information
        oldImageShape = [np.shape(self.currentImage)[0], np.shape(self.currentImage)[1]]
        print("This is the image shape: ", oldImageShape)
        oldImageCenterX = int(oldImageShape[1] / 2)
        oldImageCenterY = int(oldImageShape[0] / 2)
        newImageShape = [oldImageShape[0] * enlarge, oldImageShape[1] * enlarge]
        newImageCenterX = int(newImageShape[1] / 2)
        newImageCenterY = int(newImageShape[0] / 2)

        # expand box
        points = [[x1, y1], [x2, y2]]
        for idx, point in enumerate(points):
            oldDiffX = abs(oldImageCenterX - point[0])
            oldDiffY = abs(oldImageCenterY - point[1])
            if point[0] < oldImageCenterX:
                newX = newImageCenterX - int(oldDiffX * eta)
            else:
                newX = newImageCenterX + int(oldDiffX * eta)
            if point[1] < oldImageCenterY:
                newY = newImageCenterY - int(oldDiffY * eta)
            else:
                newY = newImageCenterY + int(oldDiffY * eta)
            points[idx] = [newX, newY]

        # put back in rect form
        return [points[0][0], points[0][1], points[1][0] - points[0][0], points[1][1] - points[0][1]]

    #     def predict(self, img):
    #         # gather image from feed, rescale, set in SSM
    #         self.currentImage = img.copy()
    #         self.originalImage = img.copy()
    #         self.__scale_image__()
    #         self.selective_search_model.setBaseImage(self.currentImage)

    #         # choose prediction accuracy or speed
    #         if self.fastPrediction:
    #             self.selective_search_model.switchToSelectiveSearchFast()
    #         else:
    #             self.selective_search_model.switchToSelectiveSearchQuality()

    #         # choose single or multiple strategies (multiple is default)
    #         if self.singleStrategy == True:
    #             self.selective_search_model.switchToSingleStrategy()
    #         self.allBoxes = self.selective_search_model.process()
    #         minSize = max(np.shape(self.currentImage)) + 1
    #         _, regions = selectivesearch.selective_search(self.currentImage, scale=400, sigma=0.9, min_size=minSize)
    #         self.allBoxes = []
    #         for region in regions:
    #             self.allBoxes.append(list(region['rect']))
    #         self.numberOfBoxesDetected = len(self.allBoxes)

    #         # predict and narrow down boxes
    #         self.__makeActualPrediction__()
    #         self.__NonMaximumSuppression__()
    #         self.__eliminateInnerBoxes__()

    #         # fix the boxes so that they are all in the right scale relative to original images
    #         self.__printBoxesToSmallerImages__()
    #         for idx,box in enumerate(self.allBoxes_Final):
    #             self.allBoxes_Final[idx] = self.__expand_bounding_box__(box)

    #         # print boxes into image and store image to object
    #         if self.printBoxesToImages:
    #             self.__printBoxesToImages__()

    def predict(self, img):
        # gather image from feed, rescale, set in SSM
        self.currentImage = img.copy()
        self.originalImage = img.copy()
        self.__scale_image__()

        # selective search
        minSize = max(np.shape(self.currentImage)) + 1
        _, regions = selectivesearch.selective_search(self.currentImage, scale=self.k, sigma=self.sigma,
                                                      min_size=minSize)
        self.allBoxes = []
        for region in regions:
            self.allBoxes.append(list(region['rect']))
        self.numberOfBoxesDetected = len(self.allBoxes)

        # predict and narrow down boxes
        self.__makeActualPrediction__()
        self.__NonMaximumSuppression__()
        self.__eliminateInnerBoxes__()

        # fix the boxes so that they are all in the right scale relative to original images
        self.__printBoxesToSmallerImages__()
        for idx, box in enumerate(self.allBoxes_Final):
            self.allBoxes_Final[idx] = self.__expand_bounding_box__(box)

            # print boxes into image and store image to object
        if self.printBoxesToImages:
            self.__printBoxesToImages__()

    def writeImageToFile(self, fullPathToImageFolder, imageName):
        pathToImage = join(fullPathToImageFolder, imageName + ".jpg")
        cv2.imwrite(pathToImage, self.currentImage)

    def __str__(self):
        buffer = "SUMMARY OF CURRENT STATE OF BOX DETECTOR:" + "\n"
        buffer += "imageScaleFactor: " + str(self.imageScaleFactor) + "\n"
        buffer += "sigma: " + str(self.sigma) + "\n"
        buffer += "k: " + str(self.k) + "\n"
        buffer += "predictionConfidence: " + str(self.predictionConfidence) + "\n"
        buffer += "IOU_Allowance: " + str(self.IOU_Allowance) + "\n"
        buffer += "numberOfBoxesDetected: " + str(self.numberOfBoxesDetected) + "\n"
        buffer += "numberOfBoxesChosen: " + str(self.numberOfBoxesChosen) + "\n"
        buffer += "Final Bounding Boxes: " + "\n"
        for idx, box in enumerate(self.allBoxes_Final):
            buffer += "Box %d: " % (idx + 1) + "[TOP_LEFT_CORNER:(%d,%d)\tBOTTOM_RIGHT_CORNER:(%d,%d)]" % (
                box[0], box[1], box[0] + box[2], box[1] + box[3]) + "\n"
        return buffer
