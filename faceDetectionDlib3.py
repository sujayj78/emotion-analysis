# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 09:32:03 2017

@author: Administrator
"""
import cv2, glob, random, sys, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
from sklearn.externals import joblib
from musicPlayer import *

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
video_capture = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Administrator\\emotionAwareMPlayer\\shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
faceDet = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("C:\\Users\\Administrator\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt_tree.xml")
#clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
path_ck="D:\\sorted_set"
path_art="D:\\arti_sorted_set"
dset="art"

def detect_faces(gray):
    
    #Open image
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        
        #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures == face2
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
            cv2.imwrite("D:\\prediction\\%s.jpg" %out) #Write image
            return out
        except:
            pass #If error, pass file
     #Increment image number
        
def get_files(emotion,path): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("%s\\%s\\*" %(path,emotion))
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised

def make_sets(path):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion,path)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                training_data.append(landmarks_vectorised) #append image array to training data list
                training_labels.append(emotions.index(emotion))
    
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                prediction_data.append(landmarks_vectorised)
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels   

def pred(clf):
    image=cv2.imread("D:\\arti_sorted_set\\happy\\test_happy_003.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_image=detect_faces(gray)
    clahe_image = clahe.apply(gray)
    landmarks_vectorised = get_landmarks(clahe_image)
    if landmarks_vectorised == "error":
        pass
    else:
        landmarks_vectorised = np.array(landmarks_vectorised).reshape((1, -1))
        emo=clf.predict(landmarks_vectorised)
    return emo

accur_lin = []
def main():
    clf=joblib.load("emo_analysis.pkl")
    print("Enter 1 to train and 2 to predict")
    x=int(input())
    if x==1:
        if dset=="ck":
            for i in range(0,3):
                print("Making sets %s" %i) #Make sets by random sampling 80/20%
                training_data, training_labels, prediction_data, prediction_labels = make_sets(path_ck)
                npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
                npar_trainlabs = np.array(training_labels)
                print("training SVM linear %s" %i) #train SVM
                clf.fit(npar_train, training_labels)
                print("getting accuracies %s" %i) #Use score() function to get accuracy
                npar_pred = np.array(prediction_data)
                pred_lin = clf.score(npar_pred, prediction_labels)
                print("linear: ", pred_lin)
                accur_lin.append(pred_lin) #Store accuracy in a list
            print("Mean value lin svm: %.3f" %np.mean(accur_lin)) #Get mean accuracy of the 10 runs
        else:
            for i in range(0,7):
                print("Making sets %s" %i) #Make sets by random sampling 80/20%
                training_data, training_labels, prediction_data, prediction_labels = make_sets(path_art)
                npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
                npar_trainlabs = np.array(training_labels)
                print("training SVM linear %s" %i) #train SVM
                clf.fit(npar_train, training_labels)
                print("getting accuracies %s" %i) #Use score() function to get accuracy
                npar_pred = np.array(prediction_data)
                pred_lin = clf.score(npar_pred, prediction_labels)
                print("linear: ", pred_lin)
                accur_lin.append(pred_lin) #Store accuracy in a list
            print("Mean value lin svm: %.3f" %np.mean(accur_lin)) #Get mean accuracy of the 10 runs   
    else:
        emo=pred(clf)
        print("emotion is %s"%emotions[int(emo)])
        recognize_emotion(emotions[emo])
    joblib.dump(clf,"emo_analysis.pkl")
    
if __name__=="__main__":
    main()
#sys.exit(0)