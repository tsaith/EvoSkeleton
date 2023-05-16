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
import cv2 as cv
#import imageio
import imageio.v2 as imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt

from skeleton_utils import re_order_indices, estimate_stats, normalize, unNormalizeData
from skeleton_utils import convert_holistic_to_skeleton_2d
from cascade_detector import CascadeDetector
from plot_utils import draw_skeleton, plot_3d_ax, adjust_figure

import mediapipe as mp
import yaml

from skyeye.utils.opencv import Webcam, wait_key
from skyeye.utils import Timer

from holistic_data import HolisticData

from file_utils import make_dir


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic_detector = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)



def draw_msg(image, msg, x, y, y_shift=20, color=(0, 255,0)):

    cv.putText(image, msg, (x, y),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    y += y_shift

    return (x, y)



def main():


    use_webcam = True # True: use video; False: use webcam

    #video_path = "videos/TurnBody.mp4"
    #video_path = "videos/Jump.mp4"
    #video_path = "videos/SquatDown.mp4" 
    video_path = "videos/WalkAround.mp4"
    webcam_device = 0 # Device ID

    '''
    source = "videos/WalkAround.mp4"
    for frame in iio.imiter(source, plugin="pyav"):
        cv.imshow("frame", frame)

        # Exit while 'q' or 'Esc' is pressed
        key = wait_key(1)
        if key == ord("q") or key == 27: break
    '''      


    frame_width = 640
    frame_height = 480

    use_V4L2 = True
    autofocus = False
    auto_exposure = True


    image_dir = "imgs"
    output_dir = "outputs"
    
    num_joints = 16

    # Set mediapipe using GPU
    with open(r'mediapipe.yaml', 'r', encoding='utf-8') as f:
        inputs = yaml.load(f, Loader=yaml.Loader)

    enable_gpu = inputs['enable_gpu']

    if use_webcam:

        webcam = Webcam()
        if webcam.is_open():
            webcam.release()

        cap = webcam.open(webcam_device, width=frame_width, height=frame_height,
            use_V4L2=use_V4L2, autofocus=autofocus, auto_exposure=auto_exposure)

    else:

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file. {video_path}")
            exit()

    
    
    # paths
    data_dic_path = './example_annot.npy'     
    model_path = './example_model.th'
    stats_path = './stats.npy'
    stats = np.load(stats_path, allow_pickle=True).item()
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
    
    # Detector
    detector = CascadeDetector()
    detector.load_model(model_path)
    detector.load_stats(stats_path)
    
    # Holistic data
    holistic_data = HolisticData()
    holistic_data.set_image_size(frame_width, frame_height)

    timer = Timer()

    frame_count = 0
    while True:

        frame_count += 1
        print("frame_count = {}".format(frame_count))

        ret, frame = cap.read()

        if not ret:
            break

        if frame is None:
            continue


        # Resize frame  
        frame = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)


        frame = cv.flip(frame, 1)
        image = frame.copy()
        image_out = frame.copy()

        # Flip the image horizontally for a later selfie-view display, and convert
        if enable_gpu == 1:
            # the BGR image to RGBA.
            image = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
        else:
            # the BGR image to RGB.
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic_detector.process(image)

        holistic_data.update(results)
        skeleton_2d = convert_holistic_to_skeleton_2d(holistic_data)

        #f = plt.figure()
        f = plt.figure(figsize=(15, 5))

        ax1 = plt.subplot(131)
        plt.title('Input image')
        ax1.imshow(image)

        ax2 = plt.subplot(132)
        plt.title('2D key-point inputs: {:d}*2'.format(num_joints))
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        draw_skeleton(ax2, skeleton_2d, gt=True)
        plt.plot(skeleton_2d[:,0], skeleton_2d[:,1], 'ro', 2)       


        timer.tic()

        pred = detector.predict(skeleton_2d)

        dt = timer.toc()
        fps = 1.0/dt
    
        ax3 = plt.subplot(133, projection='3d')
        plot_3d_ax(ax=ax3, 
                   pred=pred, 
                   elev=0,  # 10, 
                   azim=-60, # -90,
                   title='3D prediction'
                   )    
        adjust_figure(left = 0.05, 
                      right = 0.95, 
                      bottom = 0.08, 
                      top = 0.92,
                      wspace = 0.3, 
                      hspace = 0.3
                      )       

        # Save figure
        filename = f"snapshot_2d.jpg" 
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)

        plt.close(f)

        print(f"Time cost of inference is {dt} in seconds.")

        text_x = 20
        text_y = 20
        text_y_shift = 20

        msg = f"fps: {fps:.1f}"
        (text_x, text_y) = draw_msg(image_out, msg, text_x, text_y)

        # show the frame and record if the user presses a key
        cv.imshow("Win", image_out)


        # Exit while 'q' or 'Esc' is pressed
        key = wait_key(1)
        if key == ord("q") or key == 27: break



    '''
    for image_name in data_dic.keys():
    
        print(f"Process {image_name}")
    
        image_path = os.path.join(image_dir, image_name)
        img = imageio.imread(image_path)

        f = plt.figure(figsize=(9, 3))
        ax1 = plt.subplot(131)
        ax1.imshow(img)
        plt.title('Input image')
        ax2 = plt.subplot(132)
        plt.title('2D key-point inputs: {:d}*2'.format(num_joints))
        ax2.set_aspect('equal')
        ax2.invert_yaxis()

        skeleton_2d = data_dic[image_name]['p2d']

        draw_skeleton(ax2, skeleton_2d, gt=True)
        plt.plot(skeleton_2d[:,0], skeleton_2d[:,1], 'ro', 2)       
        # Nose was not used for this examplar model
    
        timer = Timer()
    
        timer.tic()
        print(f"skeleton_2d: {skeleton_2d}")
        print(f"skeleton_2d shape: {skeleton_2d.shape}")

        pred = detector.predict(skeleton_2d)

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
    ''' 

    #print(f"data_dic: {data_dic}")

    if use_webcam:

        # cleanup the camera and close any open windows
        if webcam.is_open():
            webcam.release()
    else:

        if cap.isOpened():
            cap.release()        

    cv.destroyAllWindows()


if __name__ == "__main__":

    main()