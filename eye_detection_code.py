import urllib.request
from urllib.request import urlopen
import cv2
import numpy as np
import time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
url= "http://192.168.43.1:8080/shot.jpg"
while True:
    imgResp=urlopen(url)
    img_arr = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)
    img=cv2.resize(img,(600,400))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            cv2.imshow('IPWebcam',img)
    
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
