import cv2 as cv
import os
import argparse

def video_to_images(video_path, output_dir, frame_interval=1, skip_frames=0, output_format='png'):

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    video = cv.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    # Initialize the current frame number
    current_frame = 0

    # Iterate through the video frames
    while current_frame < total_frames:
        # Read the next frame from the video
        ret, frame = video.read()

        if not ret:
            break

        # Save the current frame as an image if it's a multiple of frame_interval and is beyond the skipped frames
        if current_frame % frame_interval == 0 and current_frame >= skip_frames:
            image_path = os.path.join(output_dir, f'frame_{current_frame:04d}.{output_format}')
            cv.imwrite(image_path, frame)

        # Increment the current frame number
        current_frame += 1

    # Release the video file
    video.release()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert a video file to images.')
    parser.add_argument('video_path', type=str, help='The path to the input video file.')
    parser.add_argument('output_dir', type=str, help='The path to the output directory for images.')
    parser.add_argument('-i', '--frame_interval', type=int, default=1, help='The interval between frames to save as images. Default: 1')
    parser.add_argument('-s', '--skip_frames', type=int, default=0, help='The number of frames to skip at the beginning of the video. Default: 0')
    parser.add_argument('-f', '--output_format', type=str, choices=['jpg', 'jpeg', 'png', 'webp'], default='png', help='The output image format. Default: png')

    # Parse the arguments
    args = parser.parse_args()

    # Call the video_to_images function with the provided arguments
    video_to_images(args.video_path, args.output_dir, args.frame_interval, args.skip_frames, args.output_format)

if __name__ == '__main__':
    main()
