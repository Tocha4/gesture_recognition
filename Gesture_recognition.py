import numpy as np
import tensorflow as tf
import cv2 
import datetime

from Model_gesture import load, predict
    
saver = tf.train.import_meta_graph('../gestures/model/cnn-model.ckpt-20.meta')
now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
with tf.Session(graph=tf.get_default_graph()) as sess:
    load(saver, sess, epoch=20, path='../gestures/model/')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter(now+'.avi',fourcc, 10.0, (640,480))
    cap = cv2.VideoCapture(0)
    number = 0
    while(True):
        ret, frame = cap.read()
        cv2.rectangle(frame,(40,80),(315,340),(0,255,0),3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_to_pred = gray[40:315,80:340]
        X_test = np.array([img_to_pred.flatten()], dtype=np.uint8)
        preds = predict(sess, X_test, return_proba=True)
        value = np.argmax(preds)        
        preds_txt = ['{:3.2f}'.format(i) for i in preds[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Probability: '+str(preds_txt),(35,400), font, 0.45,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Prediction: '+str(value),(35,50), font, 0.75,(255,255,255),1,cv2.LINE_AA)  
        
        
        cv2.imshow('frame',frame)
#        out.write(frame)
        key = int(cv2.waitKey(20))
        if key == 48:
            cv2.imwrite('./img.jpg', frame)
            print(key)
        if key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    


