import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure





def plt_as_img(x,y):
    fig = Figure(figsize=(3.2, 2.4))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    
    ps = ax.plot(x.T,y.T, 'o')
    ax.legend(iter(ps), [str(i) for i in range(6)], loc=1)
    ax.axis('on')
    width, height = np.array(fig.get_size_inches() * fig.get_dpi(), dtype=np.uint32)
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return image





if __name__=='__main__':
    x = np.array([np.linspace(i,50) for i in range(6)])
    y = np.array([np.linspace(0,10) for _ in range(6)])
    image = plt_as_img(x,y)
    plt.imshow(image)







