"""
Am examplar script showing inference on the newly collected images in U3DPW. 
"""

import os
import sys
sys.path.append("../")

from libs.dataset.h36m.data_utils import unNormalizeData

import numpy as np
import cv2 as cv
import yaml
import mediapipe as mp

import matplotlib.pyplot as plt

from skyeye.pose_estimation.evo import CascadeDetector
from skyeye.pose_estimation.evo import convert_holistic_to_skeleton_2d
from skyeye.pose_estimation.evo import convert_to_skeleton_3d_h36m17p
from skyeye.pose_estimation.evo import plot_skeleton_2d, plot_skeleton_3d
from skyeye.pose_estimation.evo import make_snapshot_plot
#from skyeye.pose_estimation.evo import rotate_vector_3d, rotate_skeleton_3d
from skyeye.pose_estimation.evo import rotate_skeleton_3d
from skyeye.pose_estimation.evo import is_valid_skeleton
from skyeye.pose_estimation import Diagnostic

from skyeye.utils.matplotlib import figure_to_cv_image
from skyeye.utils.opencv import Webcam, wait_key
from skyeye.utils import Timer
from skyeye.utils.file import make_dir

from holistic_data import HolisticData

mp_holistic = mp.solutions.holistic
holistic_detector = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def main():

    use_webcam = False # True: use video; False: use webcam
    use_video = True  # True: use video; False: use webcam
    use_image = False

    # Rotation angle
    roll_angle = 0.0
    #roll_angle = -15.0

    video_path = "videos/sample_video.mp4"
    #video_path = "videos/TurnBody.mp4"
    #video_path = "videos/Jump.mp4"
    #video_path = "videos/SquatDown.mp4" 
    #video_path = "videos/WalkAround.mp4"

    output_dir = "outputs"

    image_dir = "images"
    #image_path = os.path.join(image_dir, "Collected/p1.jpg")
    #image_path = os.path.join(image_dir, "Samples/33.jpg")
    #image_path = os.path.join(image_dir, "TurnBody/frame_0090.jpg")
    #image_path = os.path.join(image_dir, "TurnBody/frame_0120.jpg")
    #image_path = os.path.join(image_dir, "SquatDown/frame_0150.jpg")
    #image_path = os.path.join(image_dir, "SquatDown/frame_0420.jpg")

    webcam_device = 0 # Device ID

    #frame_width = 500
    #frame_height = 500
    frame_width = 1000
    frame_height = 1000

    use_V4L2 = True
    autofocus = False
    auto_exposure = True

    # Create output video directory
    video_main_name = os.path.basename(video_path)
    video_main_name = os.path.splitext(video_main_name)[0]
    output_video_dir = os.path.join(output_dir, video_main_name)
    os.makedirs(output_video_dir, exist_ok=True)

    # Set mediapipe using GPU
    with open(r'inputs.yaml', 'r', encoding='utf-8') as f:
        inputs = yaml.load(f, Loader=yaml.Loader)

    use_gpu = inputs['use_gpu']

    if use_gpu == 1:
        use_gpu = True
    else:
        use_gpu = False


    if use_webcam:

        webcam = Webcam()
        if webcam.is_open():
            webcam.release()

        cap = webcam.open(webcam_device, width=frame_width, height=frame_height,
            use_V4L2=use_V4L2, autofocus=autofocus, auto_exposure=auto_exposure)
        
        num_frames = 100000000

    if use_video:

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file. {video_path}")
            exit()

        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"num_frames: {num_frames}")

    # paths
    model_path = './example_model.th'
    stats_path = './stats.npy'
    
    # Detector
    detector = CascadeDetector(use_gpu=use_gpu)
    detector.load_model(model_path)
    detector.load_stats(stats_path)
    
    # Holistic data
    holistic_data = HolisticData()
    holistic_data.set_image_size(frame_width, frame_height)

    # Diagnostic
    diag = Diagnostic()
    video_filename = os.path.basename(video_path)
    diag.init(video_filename=video_filename)

    timer = Timer()


    for iframe in range(num_frames):

        print(f"iframe: {iframe}")

        if use_image:
            frame = cv.imread(image_path)
        else:    
            ret, frame = cap.read()
            if not ret:
                break

        # Resize frame  
        frame = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)

        #frame = cv.flip(frame, 1)

        image = frame.copy()
        image_out = frame.copy()

        # Convert to RGB format
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        plt.imshow(image)

        # Save figure
        filename = f"input_image.jpg" 
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic_detector.process(image)

        holistic_data.update(results)
        skel_2d = convert_holistic_to_skeleton_2d(holistic_data)

        is_valid = is_valid_skeleton(skel_2d)
        print(f"is_valid: {is_valid}")

        if not is_valid:
            continue 

        timer.tic()

        pred = detector.predict(skel_2d)

        dt = timer.toc()
        fps = 1.0/dt

        skel_3d = pred.copy()
        skel_3d = skel_3d.reshape(-1, 3)
        skel_3d = convert_to_skeleton_3d_h36m17p(skel_3d)

        # Rotate the skeleton
        skel_3d = rotate_skeleton_3d(skel_3d, roll_angle, axis="x")

        message = f"fps: {fps:3.1f}"

        try:
            fig = make_snapshot_plot(image, skel_2d, skel_3d, message=message)
        except:
            continue

        # Export snapshot frame 
        snapshot_image = figure_to_cv_image(fig)
        diag.export_video_frame(snapshot_image)

        cv.imshow("snapshot", snapshot_image)

        # Close figure 
        plt.close(fig)

        print(f"Time cost of inference is {dt} in seconds.")

        # Exit while 'q' or 'Esc' is pressed
        key = wait_key(1)
        if key == ord("q") or key == 27: break

    if use_webcam:

        # cleanup the camera and close any open windows
        if webcam.is_open():
            webcam.release()

    if use_video:

        if cap.isOpened():
            cap.release()        

    cv.destroyAllWindows()


if __name__ == "__main__":

    main()