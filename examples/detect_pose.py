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
import yaml
import mediapipe as mp

import imageio.v2 as imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt

from skeleton_utils import re_order
from skeleton_utils import convert_holistic_to_skeleton_2d
from skeleton_utils import convert_to_skeleton_3d_h36m17p, scale_skeleton_3d_from_h36m
from cascade_detector import CascadeDetector
from plot_utils import plot_skeleton_2d
from plot_utils import draw_skeleton, plot_3d_ax, adjust_figure


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

    use_webcam = False # True: use video; False: use webcam
    use_image = True

    video_path = "videos/TurnBody.mp4"
    #video_path = "videos/Jump.mp4"
    #video_path = "videos/SquatDown.mp4" 
    #video_path = "videos/WalkAround.mp4"


    image_dir = "images"
    image_path = os.path.join(image_dir, "Samples/33.jpg")
    #image_path = os.path.join(image_dir, "TurnBody/frame_0090.jpg")
    #image_path = os.path.join(image_dir, "TurnBody/frame_0120.jpg")
    #image_path = os.path.join(image_dir, "SquatDown/frame_0150.jpg")
    #image_path = os.path.join(image_dir, "SquatDown/frame_0420.jpg")

    webcam_device = 0 # Device ID

    frame_width = 1000
    frame_height = 1000

    #frame_width = 640
    #frame_height = 480

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
        print(f"frame_count = {frame_count}")

        ret, frame = cap.read()

        if use_image:
            frame = cv.imread(image_path)

        #if not ret:
        #    break

        if frame is None:
            continue

        # Resize frame  
        frame = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)

        #frame = cv.flip(frame, 1)

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
        skel_2d = convert_holistic_to_skeleton_2d(holistic_data)

        #f = plt.figure()
        f = plt.figure(figsize=(15, 5))

        ax1 = plt.subplot(131)
        plt.title('Input image')
        ax1.imshow(image)

        ax2 = plt.subplot(132)
        plt.title('2D key-point inputs: {:d}*2'.format(num_joints))
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        plot_skeleton_2d(ax2, skel_2d)
        #draw_skeleton(ax2, skel_2d, gt=True)
        plt.plot(skel_2d[:,0], skel_2d[:,1], 'ro', 2)       


        timer.tic()

        pred = detector.predict(skel_2d)

        dt = timer.toc()
        fps = 1.0/dt

        skel_3d = pred.copy()
        skel_3d = skel_3d.reshape(-1, 3)

        skel_3d = convert_to_skeleton_3d_h36m17p(skel_3d)
        #skel_3d = scale_skeleton_3d_from_h36m(skel_3d, frame_width, frame_height)

        print(f"skel_2d shape: {skel_2d.shape}")
        print(f"skel_3d shape: {skel_3d.shape}")
        #print(f"hip_2d, hip_3d: {skel_2d[0]}, {skel_3d[0]}")
        #print(f"left_wrist_2d, left_wrist_3d: {skel_2d[13]}, {skel_3d[19]}")
        #print(f"right_wrist_2d, right_wrist_3d: {skel_2d[16]}, {skel_3d[27]}")


        ax3 = plt.subplot(133, projection='3d')
        plot_3d_ax(ax=ax3, 
                   pred=pred, 
                   elev=0,  # 10, 
                   azim=0, # -90,
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
        filename = f"snapshot.jpg" 
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)


        f = plt.figure(figsize=(15, 5))

        ax1 = plt.subplot(131)
        plt.title('Input image')
        ax1.imshow(image)

        ax2 = plt.subplot(132)
        plt.title('2D key-point inputs: {:d}*2'.format(num_joints))
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        plot_skeleton_2d(ax2, skel_2d)
        #draw_skeleton(ax2, skel_2d, gt=True)
        plt.plot(skel_2d[:,0], skel_2d[:,1], 'ro', 2)       

        skel_3d_to_2d = skel_3d[:,0:2]

        ax3 = plt.subplot(133)
        plt.title(f"skel_3d to 2d")
        ax3.set_aspect('equal')
        ax3.invert_yaxis()
        plot_skeleton_2d(ax3, skel_3d_to_2d)
        plt.plot(skel_3d_to_2d[:,0], skel_3d_to_2d[:,1], 'ro', 2)       


        # Save figure
        filename = f"snapshot_test.jpg" 
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)

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