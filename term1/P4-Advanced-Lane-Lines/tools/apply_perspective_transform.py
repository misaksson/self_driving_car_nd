"""Apply perspective transform to video

Loads an transformation matrix as provided by find_perspective_transform.py and
applies the transform to all images in video file. The original and transformed
images are then put side-by-side and written to output video file.
"""

from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import pickle


def process_image(input_image):
    h, w, _ = input_image.shape
    transformed_image = cv2.warpPerspective(input_image, transformation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return np.concatenate((input_image, transformed_image), axis=1)


with open('../output/perspective_transform.p', 'rb') as fid:
    data = pickle.load(fid)
    transformation_matrix = data['transformation_matrix']


clip = VideoFileClip('../input/project_video.mp4').fl_image(process_image)
clip.write_videofile('../output/project_video_perspective.mp4', audio=False)
clip = VideoFileClip('../input/challenge_video.mp4').fl_image(process_image)
clip.write_videofile('../output/challenge_video_perspective.mp4', audio=False)
clip = VideoFileClip('../input/harder_challenge_video.mp4').fl_image(process_image)
clip.write_videofile('../output/harder_challenge_video_perspective.mp4', audio=False)
