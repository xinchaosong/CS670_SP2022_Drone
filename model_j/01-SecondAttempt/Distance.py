#!/usr/bin/env python

#Use: Library for all kinds of distance-related operations.

#Set Up Modules:
#--------------------------------------------------------------------------------------
#common modules
import numpy as np                            #library for working with arrays
# import cv2                                    #libary for computer vision tools
# import math                                   #library for math tools
from scipy.spatial.distance import cdist      #library for distance matrix computation 

#mine
from CustomAssertions import CheckVariableIs  #libary with assertions and catch statements


#PACKAGE-STATIC FUNCTIONS:
#--------------------------------------------------------------------------------------    
def ClosestPoint(points, ref_point):
    """
        Finds point which is closest to a reference point.
        
        @param points    - list of tuples or lists - data of form [(x,y)] or [[x,y]].
        @param ref_point - tuple or list           - data of form (x,y) or [x,y].
        @return          - list[x,y]               - point in 'others' closest to 'pt'.
    """
    distances = cdist([ref_point], points)
    return points[distances.argmin()]

def FarthestPoint(points, ref_point):
    """
        Finds point which is farthest from a reference point.
        
        @param points    - list of tuples or lists - data of form [(x,y)] or [[x,y]].
        @param ref_point - tuple or list           - data of form (x,y) or [x,y].
        @return          - list[x,y]               - point in 'others' closest to 'pt'.
    """
    distances = cdist([ref_point], points)
    return points[distances.argmax()]

def EuclideanDistance(points, ref_point):
    """
        Calculates Euclidean distance between set of points and a singular reference point.
        
        @param  points    - list of tuples or lists - data of form [(x,y)] or [[x,y]].
        @param  ref_point - tuple or list           - data of form (x,y) or [x,y].
        @return dist      - list of floats          - distance between ref_point and each "points". Same order as "points".
    """
    #make sure correct inputs
    funcName = "EuclideanDistance"
    CheckVariableIs(funcName,"points").NotNone(points)
    CheckVariableIs(funcName,"ref_point").NotNone(ref_point)
    try:
        CheckVariableIs(funcName,"points").ListOfLists(points)
    except:
        CheckVariableIs(funcName,"points").ListOfTuples(points)
                
    try:
        CheckVariableIs(funcName,"ref_point").List(ref_point)
    except:
        CheckVariableIs(funcName,"ref_point").Tuple(ref_point)

    #remove any excessive, unnecessary dimensions
    points = list(np.squeeze(points))
    
    #---perform calculation all at once
    if len(np.shape(points)) == 1:
        dist = 0
        for i in range(0, len(ref_point), 1):
            dist += (points[i] - ref_point[i]) ** 2
        dist = dist ** (1 / 2)
    else:
        dist = (np.array(points) - ref_point)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
    return dist.tolist()

def MinimumDistance(points,ref_point):
    """
        Calculates smallest Euclidean distance between set of points and a single reference point.
        
        @param  points    - list of tuples or lists - data of form [(x,y)] or [[x,y]].
        @param  ref_point - tuple or list           - data of form (x,y) or [x,y].
        @return           - float                   - distance between p IN "points" and "ref_point", where p is the closest point.
    """
    #make sure correct inputs.
    funcName = "MinimumDistance"
    CheckVariableIs(funcName,"points").NotNone(points)
    CheckVariableIs(funcName,"ref_point").NotNone(ref_point)
    try:
        CheckVariableIs(funcName,"points").ListOfLists(points)
    except:
        CheckVariableIs(funcName,"points").ListOfTuples(points)
                
    try:
        CheckVariableIs(funcName,"ref_point").List(ref_point)
    except:
        CheckVariableIs(funcName,"ref_point").Tuple(ref_point)    

    return EuclideanDistance([ClosestPoint(points, ref_point)], ref_point)

def CenterOfMass(points):
    """
        Calculates center of mass for a given set of coordinates. 
        
        @param  points  - list of tuples or lists - data of form [(x,y)] or [[x,y]].
        @return         - tuple(x,y)              - center of mass of all "points".
    """
    #make sure correct inputs.
    funcName = "CenterOfMass"
    CheckVariableIs(funcName,"points").NotNone(points)
    try:
        CheckVariableIs(funcName,"points").ListOfLists(points)
    except:
        CheckVariableIs(funcName,"points").ListOfTuples(points)    
    
    #calculate
    p_size = len(points)
    x = sum([x for x,y in points])//p_size
    y = sum([y for x,y in points])//p_size
    return (x,y)

def PointsRelativeToCircle(points, center, radius):
    """
        Classify points as being inside or outside of a given circle.
        
        @param points   - list of tuples or list - data of form [(x,y)] or [[x,y]].
        @param center   - (x,y) or [x,y]         - center of a circle.  
        @param radius   - float/int              - radius of a circle.
        @return inside  - list of tuples or list - points inside circle at 'center' with 'radius'.
        @return outside - list of tuples or list - points outside circle at 'center' with 'radius'.
    """
    #make sure correct inputs.
    funcName = "PointsRelativeToCircle"
    CheckVariableIs(funcName,"points").NotNone(points)
    CheckVariableIs(funcName,"center").NotNone(center)
    CheckVariableIs(funcName,"radius").NotNone(radius)
    try:
        CheckVariableIs(funcName,"points").ListOfLists(points)
    except:
        CheckVariableIs(funcName,"points").ListOfTuples(points)
                
    try:
        CheckVariableIs(funcName,"center").List(center)
    except:
        CheckVariableIs(funcName,"center").Tuple(center) 
    
    #calculate
    radii = EuclideanDistance(points, center)
    inside = [points[x] for x in [index for index,value in enumerate(radii) if value <= radius]]
    outside = [points[x] for x in [index for index,value in enumerate(radii) if value >= radius]]
    return inside, outside

def StripInterleavedPoints(toStrip, ensureToStrip, period):
    """
        Strip points from a list of points at a uniform periodicity. Points which are required to be taken out
        will also be taken out (ensureToStrip).
        
        @param toStrip       - list - list of any kind.
        @param ensureToStrip - list - list of points which should definitely be pulled from "toStrip".
        @param period        - int  - periodicity of taking items from "toStrip".
        @return              - list - list of items from "ensureToStrip" and those in "toStrip" taken every "period" number in list.
    """
    #make sure correct inputs.
    funcName = "StripInterleavedPoints"
    CheckVariableIs(funcName,"toStrip").List(toStrip)
    CheckVariableIs(funcName,"ensureToStrip").List(ensureToStrip)
    CheckVariableIs(funcName,"period").Int(period)
    return [toStrip[i] for i in range(len(toStrip)) if (i % period == 0 or toStrip[i] in ensureToStrip)]

def ShortAdvancement(src, dst, pace=0.01):
    """
        Move a point from its current "sourcePoint" position towards
        a new "destinationPoint" by a certain amount.
    """
    x_gap, y_gap = dst[0]-src[0], dst[1]-src[1]
    return [int(src[0]+pace*x_gap), int(src[1]+pace*y_gap)]   

def ShortAdvancementArray(src, destPts, pace=0.01):
    """
        Move an array of points from current "sourcePoint" positions towards
        a new "destinationPoint" by a certain amount. Return is an array of 
        "new" destinations.
    """
    return list(map(lambda d: ShortAdvancement(src, d, pace), destPts))


#EOF