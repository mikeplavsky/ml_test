import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,resolution=0.02):
    
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min,x1_max,resolution),
        np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(
        np.array(
            [xx1.ravel(),
             xx2.ravel()]).T)   
    
    Z = Z.reshape(xx1.shape)
    
    cmap = ListedColormap(
            ("red", "blue"))
    
    plt.contourf(
        xx1,
        xx2,
        Z,
        alpha=0.4,
        cmap=cmap)
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    plt.scatter(

        x=X[:50,0],
        y=X[:50,1],

        alpha=0.8,
        c=cmap(0),

        marker="s"

    )
    
    plt.scatter(

        x=X[50:100,0],
        y=X[50:100,1],

        alpha=0.8,
        c=cmap(1),

        marker="o"

    )
    