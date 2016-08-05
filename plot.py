import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(
        X,y,classifier,
        test_idx=None,
        resolution=0.02):
    
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

    colors = (
            "red", 
            "blue",
            "lightgreen", 
            "gray",
            "cyan")
    
    cmap = ListedColormap(
            colors[:len(np.unique(y))])
    
    plt.contourf(
        xx1,
        xx2,
        Z,
        alpha=0.4,
        cmap=cmap)
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    markers = ("s", "x", "o", "^", "v")

    for i,yv in enumerate(np.unique(y)):
    
        plt.scatter(

            x=X[y == yv,0],
            y=X[y == yv,1],

            alpha=0.8,
            c=cmap(i),

            marker=markers[i]

        )

    if test_idx:
        
        X_test = X[test_idx,:]

        plt.scatter(
                x = X_test[:,0], 
                y = X_test[:,1],
                c='',
                s=55,
                marker="o",
                alpha=1.0,
                linewidth=1)
