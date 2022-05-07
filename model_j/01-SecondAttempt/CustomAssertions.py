#!/usr/bin/env python

#Use: Library to contain all common assertions with user-made assertion catches.

#Modules:
#--------------------------------------------------------------------------------------
# import numpy as np                         #library for working with arrays
# import cv2                                 #libary for computer vision tools
# import math                                #library for math tools


#PACKAGE-STATIC FUNCTIONS:
#--------------------------------------------------------
class CheckVariableIs():
    def __init__(self, callingFunction, parameterName):
        self.callingFunction = callingFunction
        self.parameterName = parameterName
    
    def NotNone(self,item):
        try:
            assert(not isinstance(item,type(None)))
        except AssertionError: 
            raise AssertionError(self.callingFunction + ": '" + self.parameterName + "' is NULL.")

    def Int(self,item):
        try:
            assert(isinstance(item,int))
        except AssertionError: 
            raise AssertionError(self.callingFunction + ": '" + self.parameterName + "' is " + str(type(item)) + " needs to be of type <int>.")
            
    def String(self,item):
        try:
            assert(isinstance(item,str))
        except AssertionError: 
            raise AssertionError(self.callingFunction + ": '" + self.parameterName + "' is " + str(type(item)) + " needs to be of type <str>.")            
            
    def List(self,item):
        try:
            assert(isinstance(item,list))
        except AssertionError: 
            raise AssertionError(self.callingFunction + ": '" + self.parameterName + "' is " + str(type(item)) + " needs to be of type <list>.")

    def Tuple(self,item):
        try:
            assert(isinstance(item,tuple))
        except AssertionError: 
            raise AssertionError(self.callingFunction + ": '" + self.parameterName + "' is " + str(type(item)) + " needs to be of type <tuple>.")
            
    def ListOfLists(self,item):
        self.List(item)
        try:
            assert(len(item)>0)
            self.List(item[0])
        except AssertionError: 
            raise AssertionError(self.callingFunction + ": '" + self.parameterName + "' is " + str(type(item)) + " but needs to be of type <list<list>>.")            
            
    def ListOfTuples(self,item):
        self.List(item)
        try:
            assert(len(item)>0)
            self.Tuple(item[0])
        except AssertionError: 
            raise AssertionError(self.callingFunction + ": '" + self.parameterName + "' is " + str(type(item)) + " but needs to be of type <list<tuple>>.")            

            
#EOF