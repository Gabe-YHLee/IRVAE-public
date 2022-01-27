import yaml
import numpy as np
from matplotlib.patches import Ellipse, Rectangle, Polygon

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def label_to_color(label):
    
    n_points = label.shape[0]
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((11, 3))
    rgb[0, :] = [253, 134, 18]
    rgb[1, :] = [106, 194, 217]
    rgb[2, :] = [111, 146, 110]
    rgb[3, :] = [153, 0, 17]
    rgb[4, :] = [179, 173, 151]
    rgb[5, :] = [245, 228, 0]
    rgb[6, :] = [255, 0, 0]
    rgb[7, :] = [0, 255, 0]
    rgb[8, :] = [0, 0, 255]
    rgb[9, :] = [18, 134, 253]
    rgb[10, :] = [155, 155, 155] # grey

    for idx_color in range(10):
        color[label == idx_color, :] = rgb[idx_color, :]
    return color

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def PD_metric_to_ellipse(G, center, scale, **kwargs):
    
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # find angle of ellipse
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # draw ellipse
    width, height = 2 * scale * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)

def rectangle_scatter(size, center, color):

    return Rectangle(xy=(center[0]-size[0]/2, center[1]-size[1]/2) ,width=size[0], height=size[1], facecolor=color)

def triangle_scatter(size, center, color):
    
    return Polygon(((center[0], center[1] + size[1]/2), (center[0] - size[0]/2, center[1] - size[1]/2), (center[0] + size[0]/2, center[1] - size[1]/2)), fc=color)
