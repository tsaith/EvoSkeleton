import numpy as np
import matplotlib.pyplot as plt

from skeleton_utils import pose_connection, re_order_indices, re_order


def plot_skeleton_2d(ax, skeleton, with_index=True):

    for connect in pose_connection:

        point1_idx = connect[0]
        point2_idx = connect[1]

        point1 = skeleton[point1_idx]
        point2 = skeleton[point2_idx]
        color = 'black'

        plt.plot([int(point1[0]),int(point2[0])], 
                 [int(point1[1]),int(point2[1])], 
                 c=color, 
                 linewidth=2)

    if with_index:
        for i, joint in enumerate(skeleton):
            plt.text(joint[0], joint[1], str(i), color=color)

    return


def draw_skeleton(ax, skeleton, gt=False, add_index=True):

    for segment_idx in range(len(pose_connection)):
        point1_idx = pose_connection[segment_idx][0]
        point2_idx = pose_connection[segment_idx][1]
        point1 = skeleton[point1_idx]
        point2 = skeleton[point2_idx]
        color = 'k' if gt else 'r'
        plt.plot([int(point1[0]),int(point2[0])], 
                 [int(point1[1]),int(point2[1])], 
                 c=color, 
                 linewidth=2)

    if add_index:
        for (idx, re_order_idx) in enumerate(re_order_indices):
            plt.text(skeleton[re_order_idx][0], skeleton[re_order_idx][1],
                str(idx+1), color='b')

    return

def show3Dpose(channels, 
               ax, 
               lcolor="#3498db", 
               rcolor="#e74c3c", 
               add_labels=True,
               gt=False,
               pred=False
               ):
    vals = np.reshape( channels, (32, -1) )
    I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
    J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        if gt or pred:
            color = 'k' if gt else 'r'
            ax.plot(x,y, z,  lw=2, c=color)        
        else:
            ax.plot(x,y, z,  lw=2, c=lcolor if LR[i] else rcolor)


    RADIUS = 750 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
    ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)

    # Get rid of the lines in 3d
    ax.xaxis.line.set_color(white)
    ax.yaxis.line.set_color(white)
    ax.zaxis.line.set_color(white)

    ax.invert_zaxis()

    return

def plot_3d_ax(ax, elev, azim, pred, title=None):

    ax.view_init(elev=elev, azim=azim)
    show3Dpose(re_order(pred), ax)     
    plt.title(title)    
    return

def adjust_figure(left = 0, right = 1, bottom = 0.01, top = 0.95,

    wspace = 0, hspace = 0.4):  
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    return
