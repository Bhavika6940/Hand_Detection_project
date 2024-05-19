import matplotlib.pyplot as plt
import cv2
import time
import os
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
from cvzone.ClassificationModule import Classifier



mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils

def mediapipe_detection(image,model): 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    image.flags.writeable=False  
    results=model.process(image)   
    image.flags.writeable=True   
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)  
    return image,results 

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS) 


def draw_style_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)
                              )
     
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10),thickness=1,circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121),thickness=1,circle_radius=2)
                              ) 
    
classifier=Classifier("Model/keras_model.h5","Model/labels.txt")
#print(dir(classifier))
print(classifier.list_labels)
#labels=["Hello","Thanks","Iloveyou"]
labels=classifier.list_labels
offset=20
imgSize=300
counter=0
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
    while cap.isOpened():

        
    

        ret, frame = cap.read()
        hands,frame=detector.findHands(frame)
        

        
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
       
        draw_style_landmarks(image,results)
        if hands:
            hand = hands[0]
            x,y,w,h= hand['bbox']
            imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
            imgCrop=frame[y-offset:y+h+offset,x-offset:x+w+offset]
            imgCropShape=imgCrop.shape
            aspectRatio=h/w


            if aspectRatio>1:
               k=imgSize/h
               wCal=math.ceil(k*w)
               imgResize= cv2.resize(imgCrop,(wCal,imgSize))
               imgResizeShape=imgResize.shape
               wGap=math.ceil((imgSize-wCal)/2)

               imgWhite[:,wGap:wCal+wGap]=imgResize
               prediction, index=classifier.getPrediction(imgWhite)
            #  print(prediction,index)
               print("Printing prediction")
            #  print(prediction)
              # print(dir(prediction))
               #print("printiing index",index)
               print(labels[index])
               time.sleep(2)


            else:
               k=imgSize/w
               hCal=math.ceil(k*h)
               imgResize= cv2.resize(imgCrop,(imgSize,hCal))
               imgResizeShape=imgResize.shape
               hGap=math.ceil((imgSize-hCal)/2)

               imgWhite[hGap:hCal+hGap,:]=imgResize
               prediction, index=classifier.getPrediction(imgWhite)
               
               
            #    print(prediction,index)
            #    print(prediction)            
               
            
            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)
            
       
        cv2.imshow('OpenCV Feed',image)
        cv2.waitKey(1)
        

     
        if cv2.waitKey(10) & 0xFF == ord('q'):  
           break
    cap.release()
    cv2.destroyAllWindows()