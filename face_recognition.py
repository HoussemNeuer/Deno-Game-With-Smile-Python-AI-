import cv2
import numpy as np
from pynput.keyboard import Key, Controller

keyboard = Controller()

recognizer = cv2.face.EigenFaceRecognizer_create(15,4000)

recognizer.read('trainer/trainingDataEigan.xml')

cascadePath = "haarcascade_frontalface_default.xml"
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)
ID= 1
while True:
    ret, im =cam.read()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    for (x, y , w ,h) in faces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,255), 1)
        
        Gray_Face = cv2.resize((gray[y: y+h ,x:x+w]),(110,110))
        
        Eyes =eye_cascade.detectMultiScale(Gray_Face)
        
        for(ex,ey,ew,eh) in Eyes:
            ID,conf = recognizer.predict(Gray_Face)
        if(ID == 1 ):
            ID = "No Smile"
        elif(ID == 2):
            ID = "No Smile"
        elif(ID == 3):
            ID = "Smile"
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            
        elif(ID == 4):
            ID = "Smile"
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        else:
            ID = ("Unknow")
            print(ID)

        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(ID), (x,y-40), font, 2, (255,255,255), 2)

    cv2.imshow('im',im) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
