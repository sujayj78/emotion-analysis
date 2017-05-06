# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:51:37 2017

@author: Administrator
"""

import pandas, random
import sys, os, subprocess
import cv2
import numpy as np

actions={}
def open_stuff(filename): #Open the file, credit to user4815162342, on the stackoverflow link in the text above
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

df = pandas.read_excel("EmotionLinks.xlsx") #open Excel file
actions["anger"] = [x for x in df.angry.dropna()] #We need de dropna() when columns are uneven in length, which creates NaN values at missing places. The OS won't know what to do with these if we try to open them.
actions["happy"] = [x for x in df.happy.dropna()]
actions["sadness"] = [x for x in df.sad.dropna()]
actions["neutral"] = [x for x in df.neutral.dropna()]

#And we alter recognize_emotion() to retrieve the appropriate action list and pick a random item:
def recognize_emotion(emo):
    predictions = []
    confidence = []
#    for x in facedict.keys():
#        pred, conf = fishface.predict(facedict[x])
#        cv2.imwrite("images\\%s.jpg" %x, facedict[x])
#        predictions.append(pred)
#        confidence.append(conf)
#    recognized_emotion = emotions[max(set(predictions), key=predictions.count)]
    recognized_emotion=emo
#    print("I think you're %s" %recognized_emotion)
    actionlist = [x for x in actions[recognized_emotion]] #<----- get list of actions/files for detected emotion
    random.shuffle(actionlist) #<----- Randomly shuffle the list
    open_stuff(actionlist[0]) #<----- Open the first entry in the list
#    print(actionlist[0])
