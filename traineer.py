import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2. face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImageWithId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)];
 
    faces=[]
    Ids=[]
    for imagePaths in imagePaths:
        faceImg = Image.open(imagePaths).convert('L');
        faceNp = np.array(faceImg,'uint8');
        Id = int(os.path.split(imagePaths)[-1].split('.')[1]);
        faces.append(faceNp);
        Ids.append(Id);
        cv2.imshow('training',faceNp);
        cv2.waitKey(10)
    return np.array(Ids),faces


Ids,faces = getImageWithId(path);
recognizer.train(faces,Ids)
recognizer.save('recognizer/trainingdata.yml')

getImageWithId(path);
cv2.destroyAllWindows();
