import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def plt_as_img(x,y):
    fig = Figure(figsize=(6.4, 4.8))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    
    ps = ax.plot(x.T,y.T, '-')
    ax.legend(iter(ps), [str(i) for i in range(6)], loc=1)
    
    ax.set_xlabel('time [seconds]')
    ax.set_ylabel('accuracy of predicted digit')
    width, height = np.array(fig.get_size_inches() * fig.get_dpi(), dtype=np.uint32)
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return image

#%% load_mnist function
def load_gestures(path='./gestures/', shuffle=True, random_seed=None):
    files = os.listdir(path)
    data_X = np.zeros((len(files), 71500), dtype=np.uint8)
    data_y = np.zeros(len(files), dtype=np.uint8)
    for n, name in enumerate(files):
        y = int(name.split('.')[0][-1])
        X = cv2.imread(os.path.join(path, name),0)
        X = np.array(X.flatten(), dtype=np.uint8)
        data_X[n] = np.array(X)
        data_y[n] = y
    
    idx = np.arange(data_y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        data_X = data_X[idx]
        data_y = data_y[idx]
        
    return data_X, data_y


#%% batch_generator function

def batch_generator(X,y, batch_size=50, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
    for i in range(0,X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i: i+batch_size])
        
        
if __name__=='__main__':
    path='../gestures/train/'
    X,y = load_gestures(path=path)
    np.save('X_train', X)
    np.save('y_train', y)
    
    X,y = load_gestures(path=path.replace('train', 'test'))
    np.save('X_test', X)
    np.save('y_test', y)
    