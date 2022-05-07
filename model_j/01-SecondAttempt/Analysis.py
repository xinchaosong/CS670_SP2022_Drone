#!/usr/bin/env python

#Use: Library for analyzing the performance of my algorithms.

#Modules:
#--------------------------------------------------------------------------------------
#common
import numpy as np                 #library for working with arrays
import time                        #library for timing tools

#mine
import Imaging
import copy


#CLASSES:
#--------------------------------------------------------------------------------------
#------------------------------------
#GestureAbacus-----------------------
#------------------------------------
class GestureAbacus:  
    def __init__(self):
        self.assessments = 0
        self.correct_assessments = 0
        self.all_gestures        = [0] * 10
        self.correct_gestures    = [0] * 10
        self.incorrect_gestures  = [0] * 10
    
    def summary(self):
        print("Total: %d, Correct: %d (%0.2f%%), R: %s, W: %s" % (self.assessments, self.correct_assessments, (self.correct_assessments/self.assessments)*100,
              self.correct_gestures, self.incorrect_gestures))

    def full_summary(self):
        print("Total Assessments: %d" % self.assessments)
        print("Total Performance: %0.2f%%" % ((self.correct_assessments/self.assessments)*100))
        
        for i in range(len(self.correct_gestures)):
            print("Gesture %d  ---> Correct: %d (%0.2f%%) " % (i+1, self.all_gestures[i], (self.correct_gestures[i]/max(1,self.all_gestures[i]))*100))

            
            
#------------------------------------
#StopWatch---------------------------
#------------------------------------        
class StopWatch:
    def __init__(self):
        self.start = 0
#         self.end   = 0
        self.elapsed = 0
        self.cum_elapsed = 0
        self.overhead = 0
    
    def Start(self):
        _o_ = time.time()
        self.start = time.time()
        self.elapsed = 0
        self.overhead += (time.time() - _o_)
        
    def Stop(self):
        _o_ = time.time()
        self.end = time.time()
        self.elapsed = self.end - self.start
        self.cum_elapsed += self.elapsed
        self.overhead += (time.time() - _o_)

    def TimeSummary(clocks, clockTitles, nullClock, numberOfAssessments):
        for i in range(len(clocks)):
            if clockTitles[i] == "overall":
                print("%s: (%0.4f seconds per image):" % (clockTitles[i], (clocks[i].cum_elapsed - clocks[i].overhead - nullClock.cum_elapsed - nullClock.overhead)/numberOfAssessments))
            else:
                print("%s: (%0.4f seconds per image):" % (clockTitles[i], (clocks[i].cum_elapsed-clocks[i].overhead)/numberOfAssessments))
            
            
#EOF