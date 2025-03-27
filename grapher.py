import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_lines(array1, array2, array3, title):
    """
    Plots two nX3 arrays as 3D lines.
    
    Parameters:
        array1 (numpy.ndarray): First nX3 array.
        array2 (numpy.ndarray): Second nX3 array.
    """
    if array1.shape[1] != 3 or array2.shape[1] != 3 or array3.shape[1] != 3:
        raise ValueError("Both input arrays must have a shape of nX3")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(array1[:, 0], array1[:, 1], array1[:, 2], label='ground truth', color='g')
    ax.plot(array2[:, 0], array2[:, 1], array2[:, 2], label='prediction', color='b')
    ax.plot(array3[:, 0], array3[:, 1], array3[:, 2], label='sensor', color='r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    print(f"Sensor ARE: {np.mean(np.abs((array3 - array1)/array1))}")
    print(f"{title} ARE: {np.mean(np.abs((array2 - array1[1:])/array1[1:]))}")

    plt.tight_layout()

    plt.title(title)
    
    return plt