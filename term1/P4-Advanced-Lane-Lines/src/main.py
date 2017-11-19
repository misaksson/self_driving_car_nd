import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle
import os

from camera_calibration import Calibrate
from extractor import Extractor
from detector import Detector, Line

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
detector = Detector()

clip_path = '../input/project_video.mp4'
clip = VideoFileClip(clip_path)
for rgb_image in clip.iter_frames():
    rgb_image = calibration.undistort(rgb_image)

    images = {
        'rgb': rgb_image,
        'gray': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY),
        'hls': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS),
        'bgr': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
    }

    # Extract possible lane lines pixels into a binary image.
    extracted_image = extractor.apply(images).astype(np.uint8)
    h, w = extracted_image.shape

    # Warp perspective to birds-eye-view
    perspective_image = cv2.warpPerspective(extracted_image, transformation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    # Fit lines to extracted lane line pixels.
    detector.find(perspective_image)
    cv2.imshow("Input", images['bgr'])
    cv2.imshow("Lines", Line.demo_image)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break
else:
    cv2.waitKey()

cv2.destroyAllWindows()
