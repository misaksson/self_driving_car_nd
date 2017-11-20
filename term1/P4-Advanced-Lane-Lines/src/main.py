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


def overlay_lane_detection(bgr_image, detector, inv_transformation_matrix):
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
    draw_image = cv2.warpPerspective(warped_draw_image, inv_transformation_matrix,
                                     (bgr_image.shape[1], bgr_image.shape[0]))
    # Blend the drawing image into the camera image.
    return cv2.addWeighted(bgr_image, 1, draw_image, 0.3, 0)


def overlay_lane_curvature(bgr_image, lines_curvature):
    """Write left and right lane lines curvature into the image"""
    left_line_str = f"Left curve: {lines_curvature[0]:4.0f} m"
    right_line_str = f"Right curve: {lines_curvature[1]:4.0f} m"
    cv2.putText(bgr_image, left_line_str, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr_image, right_line_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr_image


def overlay_lane_center_offset(bgr_image, lane_center_offset):
    """Write lane center offset into the image"""
    position_str = f"Lane center offset: {lane_center_offset:.2f} m"
    cv2.putText(bgr_image, position_str, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr_image


def overlay_extracted_image(bgr_image, extracted_image):
    """Draw extracted image as picture in picture"""
    h, w, _ = bgr_image.shape
    scaled_extracted_image = cv2.resize(extracted_image, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
    scaled_extracted_image *= 255
    x_offset = 2 * w // 4 - 40
    y_offset = 20
    bgr_image[y_offset:y_offset + scaled_extracted_image.shape[0],
              x_offset:x_offset + scaled_extracted_image.shape[1], :] = \
        np.dstack((scaled_extracted_image, scaled_extracted_image, scaled_extracted_image))
    return bgr_image


def overlay_detection_image(bgr_image, detection_image):
    """Draw detection image as picture in picture"""
    h, w, _ = bgr_image.shape
    scaled_detection_image = cv2.resize(detection_image, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
    x_offset = 3 * w // 4 - 20
    y_offset = 20
    bgr_image[y_offset:y_offset + scaled_detection_image.shape[0],
              x_offset:x_offset + scaled_detection_image.shape[1]] = scaled_detection_image
    return bgr_image


def overlay_perspective_image(bgr_image, transformation_matrix):
    """Draw perspective image as picture in picture"""
    h, w, _ = bgr_image.shape
    transformed_image = cv2.warpPerspective(bgr_image, transformation_matrix, (w, h),
                                            flags=cv2.INTER_LINEAR)
    scaled_transformed_image = cv2.resize(transformed_image, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
    x_offset = 3 * w // 4 - 20
    y_offset = h // 4 + 40
    bgr_image[y_offset:y_offset + scaled_transformed_image.shape[0],
              x_offset:x_offset + scaled_transformed_image.shape[1]] = scaled_transformed_image
    return bgr_image


class LaneFindingPipeline(object):
    def __init__(self):
        if camera_calibration_available():
            print("Loading camera calibration")
            self.calibration = load_camera_calibration()
        else:
            print("Running camera calibration")
            self.calibration = Calibrate()
            store_camera_calibration(self.calibration)

        self.extractor = Extractor(thresh=load_thresholds())
        self.perspective = load_perspective()
        self.detector = Detector(self.perspective)

    def player(self, video_path='../input/project_video.mp4'):
        for rgb_image in VideoFileClip(video_path).iter_frames():
            bgr_image = self.process_frame(rgb_image)

            cv2.imshow("Lane Detection", bgr_image)
            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break  # Quit by pressing Esc
            if k == 32:
                cv2.waitKey()  # Pause by pressing space
        else:
            cv2.waitKey()

        cv2.destroyAllWindows()

    def create_video(self, video_path='../input/project_video.mp4', output_video_path='../output/project_video.mp4'):
        clip = VideoFileClip(video_path).fl_image(self.process_frame_rgb)
        clip.write_videofile(output_video_path, audio=False)

    def process_frame(self, rgb_image):
        rgb_image = self.calibration.undistort(rgb_image)

        # Calculate images
        images = {
            'rgb': rgb_image,
            'gray': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY),
            'hls': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS),
            'bgr': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
        }

        # Extract possible lane lines pixels into a binary image.
        extracted_image = self.extractor.apply(images).astype(np.uint8)

        # Warp perspective to birds-eye-view
        h, w = extracted_image.shape
        perspective_image = cv2.warpPerspective(extracted_image, self.perspective['transformation_matrix'], (w, h),
                                                flags=cv2.INTER_LINEAR)

        # Fit lines to extracted lane line pixels and calculate curvature and vehicle lane position.
        lines_curvature, lane_center_offset = self.detector.find(perspective_image)

        # Draw information directly into the visualized image.
        images['bgr'] = self.draw_overlays(images['bgr'], lines_curvature, lane_center_offset, extracted_image)
        return images['bgr']

    def process_frame_rgb(self, rgb_image):
        """Process frame and convert output to RGB format"""
        bgr_image = self.process_frame(rgb_image)
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    def draw_overlays(self, bgr_image, lines_curvature, lane_center_offset, extracted_image):
        bgr_image = overlay_lane_detection(bgr_image, self.detector,
                                           self.perspective['inv_transformation_matrix'])
        bgr_image = overlay_lane_curvature(bgr_image, lines_curvature)
        bgr_image = overlay_lane_center_offset(bgr_image, lane_center_offset)
        bgr_image = overlay_extracted_image(bgr_image, extracted_image)
        bgr_image = overlay_detection_image(bgr_image, Line.demo_image)
        bgr_image = overlay_perspective_image(bgr_image, self.perspective['transformation_matrix'])
        return bgr_image


if __name__ == '__main__':
    LaneFindingPipeline().player()
    # LaneFindingPipeline().create_video()
