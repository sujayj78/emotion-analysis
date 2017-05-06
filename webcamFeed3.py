# -*- coding: utf-8 -*-
"""
Created on Thu May  4 07:14:04 2017

@author: Administrator
"""

import os, shutil, sys, time, re, glob
import dlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
from io import StringIO
from faceDetectionDlib3 import *

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Administrator\\emotionAwareMPlayer\\shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf=joblib.load("emo_analysis.pkl")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

if vc.isOpened(): # try to get the first frame
  rval, frame = vc.read()
  
else:
  rval = False

while rval:
  print("Frames will be taken now")
       
  detect = True
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  clahe_image = clahe.apply(gray)
  landmarks_vectorised = get_landmarks(clahe_image)
  if landmarks_vectorised == "error":
     pass
  else:
     landmarks_vectorised = np.array(landmarks_vectorised).reshape((1, -1)) 
     emo=clf.predict(landmarks_vectorised)
     print(emotions[emo])
     
  cv2.imshow("preview", frame)

  # Read in next frame
  rval, frame = vc.read()

  # Wait for user to press key. On ESC, close program
  key = cv2.waitKey(20)
  if key == 27: # exit on ESC
    break
  elif key == 115 or key == 83: # ASCII codes for s and S
    #filename = saveTestImage(img,outDir=saveDir)
    print("Image saved to ./" )
  

cv2.destroyWindow("preview")
joblib.dump(clf,"emo_analysis.pkl")