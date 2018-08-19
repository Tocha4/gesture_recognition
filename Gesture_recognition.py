import numpy as np
import tensorflow as tf
import cv2 
from time import time
from datetime import datetime
from _Functions_gesture import plt_as_img
from Modell import Modell





def show_cap(modell, show_jn_func=None):
    start = time()
    time_stamp = datetime.now().strftime('%Y%M%d%H%m%s')
    predictions = np.zeros(shape=(6,100))
    time_line = np.zeros(shape=(6,100))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(time_stamp+'.avi',fourcc, 5.0, (1280,480))
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        cv2.rectangle(frame,(40,80),(315,340),(0,255,0),3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_to_pred = gray[40:315,80:340]
        X_test = np.array([img_to_pred.flatten()], dtype=np.uint8)
        preds = modell.predict(sess, X_test, return_proba=True)
        value = np.argmax(preds)        
        preds_txt = ['{:3.2f}'.format(i) for i in preds[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Probability: '+str(preds_txt),(35,400), font, 0.45,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Prediction: '+str(value),(35,50), font, 0.75,(0,0,25500),1,cv2.LINE_AA)  
        
        # Adding a timeline graph to the video
        predictions = np.c_[predictions[:,1:], preds[0]]
        now = [time()-start for _ in range(6)]
        time_line = np.c_[time_line[:,1:], now]
        graph = plt_as_img(time_line, predictions)
        
        frame = np.append(frame, graph, axis=1)
        key = int(cv2.waitKey(20))
        if show_jn_func==None:
            cv2.imshow('frame',frame)
            out.write(frame)
            if key == 48:
                cv2.imwrite('./{}_img.jpg'.format(time_stamp), frame)
                print(key)
            if key & 0xFF == ord('q'):
                break    
        else:
            show_jn_func(frame)
            
    
    cap.release()
    cv2.destroyAllWindows()









if __name__=='__main__':
    
    modell = Modell()
    meta_last = modell.find_last()
    saver = tf.train.import_meta_graph(meta_last)
    
       
    with tf.Session(graph=tf.get_default_graph()) as sess:
        modell.load(sess,saver)
        show_cap(modell)



















