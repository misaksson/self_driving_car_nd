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
    def __init__(self, video_path='../input/project_video.mp4'):
        self.video_path = video_path
        image_height, image_width, _ = VideoFileClip(self.video_path).get_frame(0).shape
        confidence_cmap = cmap_builder('yellow', 'lime (w3c)', 'cyan')
        self.class_drawer = Drawer(bbox_settings=BBoxSettings(
                                   color=DynamicColor(cmap=confidence_cmap,
                                                      value_range=[0.5, 1.0],
                                                      colorbar=Colorbar(ticks=np.array([0.5, 0.75, 1.01]),
                                                                        pos=np.array([0.03, 0.97]),
                                                                        size=np.array([0.3, 0.01])))),
                                   inplace=False)
        self.track_drawer = Drawer(bbox_settings=BBoxSettings(color=StaticColor(),
                                                              border_thickness=2,
                                                              alpha_border=0.95,
                                                              alpha_fill=0.3),
                                   inplace=False)

        self.grid_generator = GridGenerators(image_height, image_width)
        self.classifier = Classifier(self.grid_generator, force_train=False, use_cache=True)
        self.cluster = Cluster(image_height, image_width)
        self.tracker = Tracker()

        with open(camera_calibration_path, 'rb') as fid:
            self.calibration = pickle.load(fid)

    def player(self):
        self.clip = VideoFileClip(self.video_path)
        self.clip_time = 0
        self.win_name = "Vehicle Detection"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        height, width, _ = self.clip.get_frame(0).shape
        cv2.resizeWindow(self.win_name, width, height)
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
                pause = not pause  # toggle
            if k == 83:
                one_step = True

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

        classified_objects = self.classifier.classify(frame_idx, bgr_image)
        clustered_objects, heatmap = self.cluster.cluster(classified_objects)
        tracked_objects = self.tracker.track(clustered_objects)
#        print(frame_idx)
#        return self.class_drawer.draw(bgr_image, objects=classified_objects)
        cv2.imshow("Classified objects", self.class_drawer.draw(bgr_image, objects=classified_objects))
        tracked_image = self.track_drawer.draw(bgr_image, objects=tracked_objects)

#        frame_idx = np.round(self.clip.fps * self.clip_time).astype(np.int)
#        extract_to_file(frame_idx, bgr_image, classified_objects)

#        for roi, size, _, color in self.grid_generator.get_params():
#            cv2.rectangle(tracked_image,
#                          (roi.left, roi.top),
#                          (roi.right, roi.bottom),
#                          color, 2)
#            cv2.rectangle(tracked_image,
#                          (roi.right - size.width + 1, roi.bottom - size.height + 1),
#                          (roi.right, roi.bottom),
#                          color, 2)
#        return self.drawer.draw(bgr_image, objects=classified_objects)
        return tracked_image

    def create_video(self, output_video_path='../output/project_video.mp4'):
        self.frame_idx = 0
        clip = VideoFileClip(self.video_path).fl_image(self._process_frame_for_video)
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


if __name__ == '__main__':
    VehicleDetectionPipeline().player()
    #VehicleDetectionPipeline().create_video()

