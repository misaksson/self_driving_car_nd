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
        return pickle.load(fid)


def draw_lane_detection(bgr_image):
    """Draw detected lane boundary in camera image

    Create an temporary birds-eye-view image where only the lane boundary is
    drawn as a green filled polygon. This image is then warped to camera view,
    and blended into the original camera image.

    Arguments:
        bgr_image - Camera image in BGR color space (for opencv).

    Returns:
        Camera image augmented with detected lane boundary as a green filled polygon.
    """
    # Draw the lane onto a temporary birds-eye-view warped image.
    warped_draw_image = np.zeros_like(bgr_image).astype(np.uint8)
    cv2.fillPoly(warped_draw_image, detector.get_lane_boundary(), (0, 255, 0))

    # Warp birds-eye-view warped image to camera image space using the inverse transformation matrix.
    draw_image = cv2.warpPerspective(warped_draw_image, perspective_transform['inv_transformation_matrix'],
                                     (bgr_image.shape[1], bgr_image.shape[0]))
    # Blend the drawing image into the camera image.
    return cv2.addWeighted(bgr_image, 1, draw_image, 0.3, 0)


if camera_calibration_available():
    print("Loading camera calibration")
    calibration = load_camera_calibration()
else:
    print("Running camera calibration")
    calibration = Calibrate()
    store_camera_calibration(calibration)


extractor = Extractor(thresh=load_thresholds())
perspective_transform = load_perspective_transform()
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
    perspective_image = cv2.warpPerspective(extracted_image, perspective_transform['transformation_matrix'], (w, h),
                                            flags=cv2.INTER_LINEAR)

    # Fit lines to extracted lane line pixels.
    detector.find(perspective_image)
    images['bgr'] = draw_lane_detection(images['bgr'])
    cv2.imshow("Input", images['bgr'])
    cv2.imshow("Lines", Line.demo_image)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break
else:
    cv2.waitKey()

cv2.destroyAllWindows()
