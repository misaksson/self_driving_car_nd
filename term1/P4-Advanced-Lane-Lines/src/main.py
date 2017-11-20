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
perspective_path = "./perspective_transform.p"


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


def load_perspective():
    with open(perspective_path, 'rb') as fid:
        return pickle.load(fid)


def overlay_lane_detection(bgr_image):
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
    draw_image = cv2.warpPerspective(warped_draw_image, perspective['inv_transformation_matrix'],
                                     (bgr_image.shape[1], bgr_image.shape[0]))
    # Blend the drawing image into the camera image.
    return cv2.addWeighted(bgr_image, 1, draw_image, 0.3, 0)


def overlay_lane_curvature(bgr_image, lines_curvature):
    left_line_str = f"Left curve: {lines_curvature[0]:4.0f} m"
    right_line_str = f"Right curve: {lines_curvature[1]:4.0f} m"
    cv2.putText(bgr_image, left_line_str, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr_image, right_line_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr_image


def overlay_lane_center_offset(bgr_image, lane_center_offset):
    position_str = f"Lane center offset: {lane_center_offset:.2f} m"
    cv2.putText(bgr_image, position_str, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr_image


if camera_calibration_available():
    print("Loading camera calibration")
    calibration = load_camera_calibration()
else:
    print("Running camera calibration")
    calibration = Calibrate()
    store_camera_calibration(calibration)


extractor = Extractor(thresh=load_thresholds())
perspective = load_perspective()
detector = Detector(perspective)

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
    perspective_image = cv2.warpPerspective(extracted_image, perspective['transformation_matrix'], (w, h),
                                            flags=cv2.INTER_LINEAR)

    # Fit lines to extracted lane line pixels and return real-world radius curvature in meter.
    lines_curvature, lane_center_offset = detector.find(perspective_image)
    images['bgr'] = overlay_lane_detection(images['bgr'])
    images['bgr'] = overlay_lane_curvature(images['bgr'], lines_curvature)
    images['bgr'] = overlay_lane_center_offset(images['bgr'], lane_center_offset)

    cv2.imshow("Input", images['bgr'])
    cv2.imshow("Lines", Line.demo_image)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break  # Quit by pressing Esc
    if k == 32:
        cv2.waitKey()  # Pause by pressing space
else:
    cv2.waitKey()

cv2.destroyAllWindows()
