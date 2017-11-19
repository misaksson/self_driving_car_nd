import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle
import os
import matplotlib.pyplot as plt

from camera_calibration import Calibrate
from extractor import Extractor
from detector import Detector

camera_calibration_path = "./calibration.p"
thresholds_path = "./thresholds.p"
perspective_transform_path = "./perspective_transform.p"


def camera_calibration_available():
    result = False
    if os.path.isfile(camera_calibration_path):
        result = True
    return result


def load_camera_calibration():
    with open(camera_calibration_path, 'rb') as fid:
        return pickle.load(fid)


def store_camera_calibration(calibration):
    with open(camera_calibration_path, 'wb') as fid:
        pickle.dump(calibration, fid)


def load_thresholds():
    if not os.path.isfile(thresholds_path):
        return dict()

    with open(thresholds_path, 'rb') as fid:
        thresholds = pickle.load(fid)
    print("Loaded thresholds:")
    print(thresholds)
    return thresholds


def load_perspective_transform():
    with open(perspective_transform_path, 'rb') as fid:
        data = pickle.load(fid)
        transformation_matrix = data['transformation_matrix']
    return transformation_matrix


if camera_calibration_available():
    print("Loading camera calibration")
    calibration = load_camera_calibration()
else:
    print("Running camera calibration")
    calibration = Calibrate()
    store_camera_calibration(calibration)


extractor = Extractor(thresh=load_thresholds())
transformation_matrix = load_perspective_transform()


clip_path = '../input/project_video.mp4'
clip = VideoFileClip(clip_path)
rgb_image = clip.get_frame(20)
#cv2.imshow('input', rgb_image)
rgb_image = calibration.undistort(rgb_image)
#cv2.imshow('undistort', rgb_image)

images = {
    'rgb': rgb_image,
    'gray': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY),
    'hls': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS),
    'bgr': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
}


lane_image = extractor.apply(images).astype(np.uint8)
#cv2.imshow('lane extracted', lane_image * 255)

h, w = lane_image.shape
perspective_image = cv2.warpPerspective(lane_image, transformation_matrix, (w, h), flags=cv2.INTER_LINEAR)
#cv2.imshow('perspective', perspective_image * 255)

detector = Detector()
detector.find(perspective_image)
cv2.waitKey()
#plt.show()
cv2.destroyAllWindows()
