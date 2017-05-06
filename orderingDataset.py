# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:11:12 2017

@author: Administrator
"""
import cv2
import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("D:\\datasets\\Emotion\\*") #Returns a list of all folders with participant numbers
faceDet = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt_tree.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert image to grayscale    
    #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures = face2
    elif len(face3) == 1:
        facefeatures = face3
    elif len(face4) == 1:
        facefeatures = face4
    else:
        facefeatures = ""
        
        #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        gray = gray[y:y+h, x:x+w] #Cut the frame to size      
        try:
            out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
        except:
            pass #If error, pass file
    return out
    
for x in participants:
    part = "%s" %x[-4:] #store current participant number
    print(part);
    for sessions in glob.glob("%s\\*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s\\*" %sessions):
            current_session = files[20:-30]
            print(current_session)
            file = open(files, 'r')
            
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            print(emotion)
            sourcefile_emo = glob.glob("D:\\datasets\\Images\\%s\\*" %current_session) #get path for last image in sequence, which contains the emotion
            #sourcefile_neutral = glob.glob("D:\\datasets\\cohn-kanade-images\\%s\\%s\\*" %(part, current_session)) #do same for neutral image
            sourcefile_emotion=sourcefile_emo[-1]
            sourcefile_neutral=sourcefile_emo[0]
            print(sourcefile_neutral)
            dest_neut = "D:\\sorted_zet\\neutral\\%s" %sourcefile_neutral[28:] #Generate path to put neutral image
            dest_emot = "D:\\sorted_zet\\%s\\%s" %(emotions[emotion], sourcefile_emotion[28:]) #Do same for emotion containing image
            image=cv2.imread(sourcefile_emotion)
            im=detect_faces(image)
            cv2.imwrite(dest_emot,im)
            image=cv2.imread(sourcefile_neutral)
            im=detect_faces(image)
            cv2.imwrite(dest_neut,im)

#            copyfile(sourcefile_neutral, dest_neut) #Copy file
#            copyfile(sourcefile_emotion, dest_emot) #Copy file
print("done")