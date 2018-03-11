import numpy as np
import tensorflow as tf
import os
import cv2 
import datetime

from Model_gesture import build_cnn, load, predict
from Functions_gesture import load_gestures



g2 = tf.Graph()
with g2.as_default():
    build_cnn(learning_rate=1e-4)
    saver = tf.train.Saver()
now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
with tf.Session(graph=g2) as sess:
    load(saver, sess, epoch=20, path='./model/')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(now+'.avi',fourcc, 10.0, (640,480))
    cap = cv2.VideoCapture(0)
    number = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.rectangle(frame,(40,80),(315,340),(0,255,0),3)
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_to_pred = gray[40:315,80:340]
        X_test = np.array([img_to_pred.flatten()], dtype=np.uint8)
        preds = predict(sess, X_test, return_proba=True)
        value = np.argmax(preds)        
        preds_txt = ['{:3.2f}'.format(i) for i in preds[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Probability: '+str(preds_txt),(35,400), font, 0.45,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Prediction: '+str(value),(35,50), font, 0.75,(255,255,255),1,cv2.LINE_AA)  
        
        
        
        
        # Display the resulting frame
        cv2.imshow('frame',frame)
        out.write(frame)
        key = int(cv2.waitKey(20))
        if key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

del g2
