import os
import sqlite3
from sqlite3 import Error
import numpy as np
import cv2
from PIL import Image





def genarate_dataset(img,id,img_id):

    cv2.imwrite("dataSet/user."+str(id)+"."+str(img_id)+".jpg",img);




def draw_boundary(img, classifier, scaleFactor, minNaighbore, color, text):
    gray_ima = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_ima, scaleFactor, minNaighbore)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color,2)
        cv2.putText(img,text,(x , y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords=[x, y, w, h]
    return coords





def detect(img, faceCascade,id,img_id):
    color ={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0)}
    coords = draw_boundary(img,faceCascade,1.1,10,color['blue'],"face");

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]];

        genarate_dataset(roi_img,id,img_id)

       # coords = draw_boundary(roi_img, eyeCascade, 1.1, 14, color['red'], "eye");
        #coords = draw_boundary(roi_img, noseCascade, 1.1, 6, color['green'], "nose");
    return img;


cap = cv2.VideoCapture(0);
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
#noseCascade = cv2.CascadeClassifier("Nariz.xml")

id = input('Enter User id:')

img_id = 0


while (cap.isOpened()):
    ret ,img =cap.read();

    img = detect(img, faceCascade,id,img_id)
    cv2.imshow('Video', img);
    img_id +=1


    if cv2.waitKey(1) & 0xFF == ord('n'):
        break;
    if img_id>100:
        break;


cap.release();

cv2.destroyAllWindows();
