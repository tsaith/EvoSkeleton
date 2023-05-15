"""
Am examplar script showing inference on the newly collected images in U3DPW. 
"""

import os
import sys
sys.path.append("../")

import libs.model.model as libm
from libs.dataset.h36m.data_utils import unNormalizeData

import torch
import numpy as np
#import imageio
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from skeleton_utils import re_order_indices, normalize, unnormalize
from plot_utils import draw_skeleton, plot_3d_ax, adjust_figure

from skyeye.utils import Timer


def get_pred(cascade, data):
    """
    Get prediction from a cascaded model
    """
    # forward pass to get prediction for the first stage
    num_stages = len(cascade)
    # for legacy code that does not have the num_blocks attribute
    for i in range(len(cascade)):
        cascade[i].num_blocks = len(cascade[i].res_blocks)
    prediction = cascade[0](data)
    # prediction for later stages
    for stage_idx in range(1, num_stages):
        prediction += cascade[stage_idx](data)
    return prediction


def main():

    image_dir = "imgs"
    output_dir = "outputs"
    
    
    num_joints = 16
    gt_3d = False  
    
    # paths
    data_dic_path = './example_annot.npy'     
    model_path = './example_model.th'
    stats = np.load('./stats.npy', allow_pickle=True).item()
    dim_used_2d = stats['dim_use_2d']
    mean_2d = stats['mean_2d']
    std_2d = stats['std_2d'] 
    # load the checkpoint and statistics
    ckpt = torch.load(model_path)
    data_dic = np.load(data_dic_path, allow_pickle=True).item()
    # initialize the model
    cascade = libm.get_cascade()
    input_size = 32
    output_size = 48
    for stage_id in range(2):
        # initialize a single deep learner
        stage_model = libm.get_model(stage_id + 1,
                                     refine_3d=False,
                                     norm_twoD=False, 
                                     num_blocks=2,
                                     input_size=input_size,
                                     output_size=output_size,
                                     linear_size=1024,
                                     dropout=0.5,
                                     leaky=False)
        cascade.append(stage_model)
    
    cascade.load_state_dict(ckpt)
    cascade.eval()
    # process and show total_to_show examples
    count = 0
    total_to_show = 10
    
    
    for image_name in data_dic.keys():
    
        print(f"Process {image_name}")
    
        image_path = os.path.join(image_dir, image_name)
        #image_path = './imgs/' + image_name
    
        img = imageio.imread(image_path)
        f = plt.figure(figsize=(9, 3))
        ax1 = plt.subplot(131)
        ax1.imshow(img)
        plt.title('Input image')
        ax2 = plt.subplot(132)
        plt.title('2D key-point inputs: {:d}*2'.format(num_joints))
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        skeleton_pred = None
        skeleton_2d = data_dic[image_name]['p2d']
        # The order for the 2D keypoints is:
        # 'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 
        # 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder'
        # 'RElbow', 'RWrist'
        draw_skeleton(ax2, skeleton_2d, gt=True)
        plt.plot(skeleton_2d[:,0], skeleton_2d[:,1], 'ro', 2)       
        # Nose was not used for this examplar model
    
        timer = Timer()
    
        timer.tic()
        norm_ske_gt = normalize(skeleton_2d, re_order_indices).reshape(1,-1)
        pred = get_pred(cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))      
        pred = unnormalize(pred.data.numpy(),
            stats['mean_3d'],
            stats['std_3d'],
            stats['dim_ignore_3d'])      
    
        dt = timer.toc()
    
        print(f"Time cost of inference is {dt} in seconds.")
    
    
        ax3 = plt.subplot(133, projection='3d')
        plot_3d_ax(ax=ax3, 
                   pred=pred, 
                   elev=10., 
                   azim=-90,
                   title='3D prediction'
                   )    
        adjust_figure(left = 0.05, 
                      right = 0.95, 
                      bottom = 0.08, 
                      top = 0.92,
                      wspace = 0.3, 
                      hspace = 0.3
                      )       
        count += 1       
        if count >= total_to_show:
            break
    
        # Save figure
        filename = f"pose_2d_{image_name}.jpg" 
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)


if __name__ == "__main__":

    main()