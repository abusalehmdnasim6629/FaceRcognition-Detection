import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainingdata.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
# font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        if Id == 6629:
            Id = 'Nasim'
        if Id == 6616:
            Id = 'Mukit'
        if Id == 6644:
            Id='Nazmul'
        if Id == 5287:
            Id= 'Hasib'
        if Id == 6654:
            Id = 'Sunny'
        if Id==6641:
            Id = 'Partho'


        cv2.putText(im,str(Id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2);
    cv2.imshow('im',im)
    if cv2.waitKey(1) & 0xFF == ord('n'):
        break
cam.release()
cv2.destroyAllWindows()
