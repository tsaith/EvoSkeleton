import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
import argparse

def images_to_video(dir_path, video_path, img_formats=['jpg', 'jpeg', 'png'],
    fps=30, repeat_frames=0):

    # Get all images in the directory with the specified formats
    images = sorted([f for f in os.listdir(dir_path) if any(f.lower().endswith(fmt) for fmt in img_formats)])

    # Read the first image to determine the resolution
    first_image_path = os.path.join(dir_path, images[0])
    first_image = cv.imread(first_image_path)
    resolution = (first_image.shape[1], first_image.shape[0])

    # Initialize the video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(video_path, fourcc, fps, resolution)

    # Add blank frames at the beginning of the video
    for _ in range(repeat_frames):
        video_writer.write(first_image)

    # Iterate through each image and write it to the video
    for image_name in tqdm(images, desc="Converting images to video"):
        image_path = os.path.join(dir_path, image_name)
        image = cv.imread(image_path)

        # Resize the image to the specified resolution
        resized_image = cv.resize(image, resolution)

        # Write the resized image to the video
        video_writer.write(resized_image)

    # Release the video writer
    video_writer.release()

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert images to a video file.')
    parser.add_argument('dir_path', type=str, help='The path to the directory containing the images.')
    parser.add_argument('video_path', type=str, help='The path to the output MP4 video.')
    parser.add_argument('-f', '--img_formats', type=str, nargs='+', default=['jpg', 'jpeg', 'png'], help='The image formats to read, separated by spaces. Default: jpg jpeg png')
    parser.add_argument('-fps', type=int, default=30, help='The frame rate of the video. Default: 30')
    parser.add_argument('-repeat', '--repeat_frames', type=int, default=0, help='The number of repeated frames used to add at the beginning of the video. Default: 0')

    # Parse the arguments
    args = parser.parse_args()

    # Call the images_to_video function with the provided arguments
    images_to_video(args.dir_path, args.video_path, args.img_formats, args.fps, args.repeat_frames)

if __name__ == '__main__':
    main()
