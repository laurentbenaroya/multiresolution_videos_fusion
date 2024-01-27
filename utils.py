import cv2
import numpy as np
import os

"""
display videos side by side
"""


def display_videos(video_list, fps=25):
    # Open the video files
    cap_list = [cv2.VideoCapture(video) for video in video_list]
    frame_count = 0

    # hard coded, create output/images and output/videos folders
    os.makedirs('output/images', exist_ok=True)
    # empty output/images folder
    for img in os.listdir('output/images'):
        os.remove(f'output/images/{img}')
    os.makedirs('output/videos', exist_ok=True)

    while True:
        # Read a frame from each video
        ret_list = []
        frame_list = []
        for cap in cap_list:
            ret, frame = cap.read()
            ret_list.append(ret)
            frame_list.append(frame)

        # If any of the reads failed, break out of the while loop
        if not all(ret_list):
            break

        # concatenate frames from each video horizontally
        all_videos = np.hstack(frame_list)

        cv2.imwrite(f'output/images/frame_{frame_count}.png', all_videos)

        frame_count += 1

    # Release the video files and close all windows
    for cap in cap_list:
        cap.release()

    # convert images to video
    # expects that ffmpeg is installed in /usr/bin
    cmd = f'/usr/bin/ffmpeg -i output/images/frame_%d.png -y -hide_banner -loglevel panic -c:v libx264 -r {fps} output/videos/multiview_video.mp4'
    print(cmd)
    os.system(cmd)


def video2frames(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # hard coded, create output/images and output/videos folders
    os.makedirs(output_path, exist_ok=True)
    # empty output/images folder
    for img in os.listdir(output_path):
        os.remove(f'{output_path}/{img}')

    while True:
        # Read a frame from each video
        ret, frame = cap.read()

        # If any of the reads failed, break out of the while loop
        if not ret:
            break

        cv2.imwrite(f'{output_path}/{frame_count}.png', frame)

        frame_count += 1

    # Release the video files and close all windows
    cap.release()


if __name__ == "__main__":
    display_videos(['video/Attal_wav2lip_gan.mp4', 'video/Attal.mp4'])