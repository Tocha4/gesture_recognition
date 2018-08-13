import os
import numpy as np
import cv2
import pandas as pd
import datetime


#%%
def create_real_imgs():
    cap = cv2.VideoCapture(0)
    
    while True:
        
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame,(40,80),(315,340),(0,255,0),3)
        cv2.imshow('frame', frame)
        
        
        key = int(cv2.waitKey(1))
        
        if key in [48,49,50,51,52,53]:
            
            now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            
            cv2.imwrite('gestures_test/{}_{}.png'.format(now, key-48), gray[40:315,80:340])
        elif key & 0xFF == ord('q'):
            print(key)
            break
        
        
    cap.release()
    cv2.destroyAllWindows()

#%%
def create_real_images_fast():
    
    cap = cv2.VideoCapture(0)
    number = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.rectangle(frame,(40,80),(315,340),(0,255,0),3)
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
        
        key = int(cv2.waitKey(1))
        
        if key in [48,49,50,51,52,53]:
            
            now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+str(number)
            print(key, now)
            cv2.imwrite('gestures_test/{}_{}.png'.format(now, key-48), gray[40:315,80:340])
            number += 1
        if key & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
#%%
def manipulating_images(path = './gestures/'):
    
    files = os.listdir(path)

    for name in files:
        y = name.split('.')[0][-1]
        img = cv2.imread(os.path.join(path, name), 0)
        rows,cols = img.shape
        
        number = 0
        for rotation in np.linspace(-25,25,10):
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
            dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
            now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%MS"))+str(number)
            cv2.imwrite('gestures_test/{}_{}.png'.format(now, y), dst)
            number += 1
        
        number = 11
        for s in np.linspace(-50,50,11):
            M = np.float32([[1,0,s],[0,1,0]])
            dst = cv2.warpAffine(img, M, (cols, rows))
            now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%MS"))+str(number)
            cv2.imwrite('gestures_test/{}_{}.png'.format(now, y), dst)
            number += 1
        
        number = 22
        for h in np.linspace(-50,50,11):
            M = np.float32([[1,0,0],[0,1,h]])
            dst = cv2.warpAffine(img, M, (cols, rows))
            now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%MS"))+str(number)
            cv2.imwrite('gestures_test/{}_{}.png'.format(now, y), dst)
            number += 1

#%%
            
if __name__=='__main__':
    
    create_real_images_fast()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
