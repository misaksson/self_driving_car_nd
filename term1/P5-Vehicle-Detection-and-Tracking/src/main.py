import cv2
import numpy as np
from colormap import cmap_builder
from moviepy.editor import VideoFileClip
from collections import namedtuple
import pickle

from grid_generator import GridGenerators
from classifier import Classifier
from cluster import Cluster
from tracker import Tracker
from drawer import *

# Load calibration from P4-Advanced-Lane-Lines
camera_calibration_path = "../../P4-Advanced-Lane-Lines/src/calibration.p"


class VehicleDetectionPipeline(object):
    def __init__(self, input_video_path='../input/project_video.mp4', output_video_path=None):
        self.input_video_path = input_video_path
        image_height, image_width, _ = VideoFileClip(self.input_video_path).get_frame(0).shape
        self.drawer = Drawer(bbox_settings=BBoxSettings(color=StaticColor(),
                                                        border_thickness=2,
                                                        alpha_border=0.95,
                                                        alpha_fill=0.3),
                             inplace=False)

        if output_video_path is None:
            show_display = True
        else:
            show_display = False

        self.grid_generator = GridGenerators(image_height, image_width)
        self.classifier = Classifier(self.grid_generator, force_train=False, use_cache=True, show_display=show_display)
        self.cluster = Cluster(image_height, image_width, show_display=show_display)
        self.tracker = Tracker(show_display=show_display)

        with open(camera_calibration_path, 'rb') as fid:
            self.calibration = pickle.load(fid)

        if output_video_path is None:
            self.player()
        else:
            self.create_video(output_video_path=output_video_path)

    def player(self):
        self.clip = VideoFileClip(self.input_video_path)
        self.clip_time = 0
        self.win_name = "Vehicle Detection"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 1200, 670)
        n_frames = np.round(self.clip.fps * self.clip.duration).astype(np.int)
        cv2.createTrackbar('frame', self.win_name, 0, n_frames - 1, self._frame_slider_callback)
        pause = False
        one_step = False
        while 1:
            if (one_step or not pause) and (self.clip_time < self.clip.duration):
                self._read_and_process_frame()
                trackbar_pos = np.round(self.clip.fps * self.clip_time).astype(np.int)
                cv2.setTrackbarPos('frame', self.win_name, trackbar_pos)
                self.clip_time += 1.0 / self.clip.fps
                one_step = False

            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break  # Quit by pressing Esc
            if k == 32:
                pause = not pause  # toggle with space key
            if k == 13:
                one_step = True  # step one frame with Enter key

        cv2.destroyAllWindows()

    def _frame_slider_callback(self, value):
        self.clip_time = value / self.clip.fps

    def _read_and_process_frame(self):
        frame_idx = np.round(self.clip.fps * self.clip_time).astype(np.int)
        rgb_image = self.clip.get_frame(self.clip_time)
        bgr_image = self._process_frame(frame_idx, rgb_image)
        cv2.imshow(self.win_name, bgr_image)

    def _process_frame(self, frame_idx, rgb_image):
        rgb_image = self.calibration.undistort(rgb_image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        medium_conf_obj, high_conf_obj, classifier_image = self.classifier.classify(frame_idx, bgr_image)
        clustered_objects, cluster_image = self.cluster.cluster(medium_conf_obj)
        tracked_objects, tracker_image = self.tracker.track(bgr_image, clustered_objects, high_conf_obj)
        output_image = self.drawer.draw(bgr_image, objects=tracked_objects)
        output_image = picture_in_picture(output_image, classifier_image, grid_pos=(0, 0), title="Classification")
        output_image = picture_in_picture(output_image, cluster_image, grid_pos=(1, 0), title="Clustering")
        output_image = picture_in_picture(output_image, tracker_image, grid_pos=(2, 0), title="Tracking")

#        frame_idx = np.round(self.clip.fps * self.clip_time).astype(np.int)
#        extract_to_file(frame_idx, bgr_image, classified_objects)

#        for roi, size, _, color in self.grid_generator.get_params():
#            cv2.rectangle(output_image,
#                          (roi.left, roi.top),
#                          (roi.right, roi.bottom),
#                          color, 2)
#            cv2.rectangle(output_image,
#                          (roi.right - size.width + 1, roi.bottom - size.height + 1),
#                          (roi.right, roi.bottom),
#                          color, 2)
        return output_image

    def create_video(self, output_video_path='../output/project_video.mp4'):
        self.frame_idx = 0
        clip = VideoFileClip(self.input_video_path).fl_image(self._process_frame_for_video)
        clip.write_videofile(output_video_path, audio=False)

    def _process_frame_for_video(self, rgb_image):
        """Process frame and convert output to RGB format"""
        self.frame_idx += 1
        bgr_image = self._process_frame(self.frame_idx, rgb_image)
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def extract_to_file(frame_idx, bgr_image, objects):
    import os
    for obj_idx, obj in enumerate(objects):
        patch = cv2.resize(bgr_image[obj.search_window.top:obj.search_window.bottom,
                                     obj.search_window.left:obj.search_window.right], (64, 64))
        file_path = os.path.join(f"../output/hard_negative_mining/frame{frame_idx:04d}_patch{obj_idx}_{int(obj.probability * 100)}.png")
        print("Writing image to ", file_path)
        cv2.imwrite(file_path, patch)


def picture_in_picture(picture, overlay, grid_pos=(0, 0), title=None):
    """Draw a picture in picture"""
    h, w, _ = picture.shape
    scaled_image = cv2.resize(overlay, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_CUBIC)
    x_offset = grid_pos[0] * w // 3
    y_offset = grid_pos[1] * h // 3
    picture[y_offset:y_offset + scaled_image.shape[0],
            x_offset:x_offset + scaled_image.shape[1]] = scaled_image
    if title is not None:
        cv2.putText(picture, title, (x_offset + 11, y_offset + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(picture, title, (x_offset + 10, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return picture


if __name__ == '__main__':
    VehicleDetectionPipeline(output_video_path='../output/project_video.mp4')
