# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:00:28 2017

@author: Administrator
"""

from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import cv2
import numpy as np
import threading
import time
import queue
from webcamFeed4 import *

running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
q1 = queue.Queue()
#q2=queue.Queue()
q_flag=False

def grab(cam, queue1,width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while(running):
        q_flag=False
        frame = {}
        frame2= {}        
        capture.grab()
        retval, img = capture.retrieve(0)
        img2=emo_det(retval,img)
        frame["img"] = img2

#        print("wow %s",emo)
#        frame2["img"]=img2
        if queue1.qsize() < 10:
            queue1.put(frame)
        else:
            print(queue1.qsize())
#        if queue2.qsize()<10:
#            queue2.put(frame2)
#        else:
#            print(queue2.qsize())
            
#def grab_2(cam, queue1, width, height, fps):
#    global running
#    capture = cv2.VideoCapture(cam)
#    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#   capture.set(cv2.CAP_PROP_FPS, fps)

#    while(running):
#        q_flag=True
#       frame = {}        
#       capture.grab()
#        retval, img = capture.retrieve(0)
#        emo=emo_det(retval,img)
#        frame["img"] = emo
#        print("wow %s",emo)

 #       if queue1.qsize() < 10:
 #           queue1.put(frame)
 #       else:
 #           print(queue1.qsize())
            
class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()



class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.startButton.clicked.connect(self.start_clicked)
        
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)
#        self.label= QtWidgets.QLabel(self.label)
#        self.ImgWidget_2 = OwnImageWidget(self.ImgWidget_2)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)


    def start_clicked(self):
        global running
        running = True
        capture_thread.start()
#        capture_thread_2.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')


    def update_frame(self):
        if not q1.empty() :
            self.startButton.setText('Camera is live')
            frame1 = q1.get()
#            frame2= q2.get()
            img1 = frame1["img"]
#            img2= frame2["img"]
#            self.label.setText(emo)
            img_height, img_width, img_colors = img1.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1
            
            img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            height, width, bpc = img1.shape
            bpl = bpc * width
            image1 = QtGui.QImage(img1.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            
            self.ImgWidget.setImage(image1)
            
#            img_height, img_width, img_colors = img2.shape
#            scale_w = float(self.window_width) / float(img_width)
#            scale_h = float(self.window_height) / float(img_height)
#            scale = min([scale_w, scale_h])

#            if scale == 0:
#                scale = 1
#            
#            img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
#            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#            height, width, bpc = img2.shape
#            bpl = bpc * width
#           image2 = QtGui.QImage(img2.data, width, height, bpl, QtGui.QImage.Format_RGB888)
#           self.ImgWidget_2.setImage(image2)

            
            

    def closeEvent(self, event):
        global running
        running = False



capture_thread = threading.Thread(target=grab, args = (0, q1, 1920, 1080, 30))
#capture_thread_2 = threading.Thread(target=grab_2, args = (0, q2, 1920, 1080, 30))

app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('Emotion Analysis')
w.show()
app.exec_()