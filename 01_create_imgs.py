import numpy as np
import cv2
import datetime


def create_real_images_fast():
    
    cap = cv2.VideoCapture(0)
    number = 0
    while(True):
        ret, frame = cap.read()
        cv2.rectangle(frame,(40,80),(315,340),(0,255,0),3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)        
        key = int(cv2.waitKey(1))        
        if key in [48,49,50,51,52,53]:
            now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+str(number)
            print(key, now)
            cv2.imwrite('../gestures/train/{}_{}.png'.format(now, key-48), gray[40:315,80:340])
            number += 1
        if key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

            
if __name__=='__main__':
    
    create_real_images_fast()
    
    
