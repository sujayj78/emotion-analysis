# -*- coding: utf-8 -*-
"""
Created on Thu May  4 07:14:04 2017

@author: Administrator
"""

import os, shutil, sys, time, re, glob
import dlib
import itertools
import operator
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
from io import StringIO
from faceDetectionDlib3 import *
from musicPlayer import *

def cvCombineTwoImages(img1,img2,buf=2,maxSize=True):
  h1, w1, c1 = img1.shape
  h2, w2, c2 = img2.shape
  
  screenWidth = 1920 # Width in pixels for macbook pro is 2880
  margin = 40 # Minimum number of extra pixels to save

  excess = w1 + w2 + buf - screenWidth + margin
  if maxSize and excess > 0:
    diff = int(np.ceil(float(excess)/4.0))

    img1 = img1[:,diff:-diff,:]
    img2 = img2[:,diff:-diff,:]

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

  h = max(h1,h2)
  w = w1 + w2 + buf
  c = max(c1,c2)

  if c1 != c2:
    # Incompatible dimensions
    print("Error, images have imcompatible dimensions along depth axis")
    return None

  img = np.zeros([h,w,c]).astype(np.uint8)

  # Add in the two images
  img[0:h1,0:w1,:] = img1
  img[0:h2,w1+buf:w1+buf+w2,:] = img2

  # Returned combined image as numpy array of uint8's
  return img

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Administrator\\emotionAwareMPlayer\\shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf=joblib.load("emo_analysis.pkl")
plotSideBySide = True # Plot before/after images together?
#cv2.namedWindow("preview")
#vc = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
limit=-1
limitList=[]
#if vc.isOpened(): # try to get the first frame
#  rval, frame = vc.read()
  
#else:
#  rval = False
def emo_det(rval,frame):
    if rval:
        print("Frames will be taken now")
#  limit=limit+1    
        frame = np.fliplr(frame)
        detect = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        landmarks_vectorised = get_landmarks(clahe_image)
        oldFrame = frame.copy()
        if landmarks_vectorised == "error":
            pass
        else:
            landmarks_vectorised = np.array(landmarks_vectorised).reshape((1, -1)) 
            emo=clf.predict(landmarks_vectorised)
            print(emotions[emo])
            
        if plotSideBySide:
            img = cvCombineTwoImages(oldFrame,frame)
            return(img)
        else:
            img = frame.copy()
            return(img)
          
#     limitList.append(emotions[emo])
     
#      cv2.imshow("preview", frame)

  # Read in next frame
#      rval, frame = vc.read()

  # Wait for user to press key. On ESC, close program
#      key = cv2.waitKey(20)
#      if key == 27: # exit on ESC
#      break
#      elif key == 115 or key == 83: # ASCII codes for s and S
    #filename = saveTestImage(img,outDir=saveDir)
#        print("Image saved to ./" )
#  elif limit==10:
#    break   
#    cv2.destroyWindow("preview")
#limitMp3=most_common(limitList)
#    joblib.dump(clf,"emo_analysis.pkl")
#recognize_emotion(limitMp3)
