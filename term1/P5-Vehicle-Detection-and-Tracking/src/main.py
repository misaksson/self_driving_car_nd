import cv2
import numpy as np
from colormap import cmap_builder
from moviepy.editor import VideoFileClip
from collections import namedtuple

from grid_generator import GridGenerators
from classifier import Classifier
from cluster import Cluster
from tracker import Tracker
from draw import Draw


ClassifiedObject = namedtuple('ClassifiedObject', ['search_window', 'confidence'])


class VehicleDetectionPipeline(object):
    def __init__(self, video_path='../input/project_video.mp4'):
        self.video_path = video_path
        image_height, image_width, _ = VideoFileClip(self.video_path).get_frame(0).shape
        confidence_cmap = cmap_builder('yellow', 'lime (w3c)', 'cyan')
        self.confidence_range = np.array([0.0, 5.0])
        self.draw = Draw(cmap=confidence_cmap, value_range=self.confidence_range)
        self.search_windows = GridGenerators(image_height, image_width)
        self.classifier = Classifier(force_train=False)
        self.cluster = Cluster(image_height, image_width)
        self.tracker = Tracker()

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
        while 1:
            if not pause and (self.clip_time < self.clip.duration):
                self._read_and_process_frame()
                frame_idx = np.round(self.clip.fps * self.clip_time).astype(np.int)
                cv2.setTrackbarPos('frame', self.win_name, frame_idx)
                self.clip_time += 1.0 / self.clip.fps

            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break  # Quit by pressing Esc
            if k == 32:
                pause = not pause  # toggle

        cv2.destroyAllWindows()

    def _frame_slider_callback(self, value):
        self.clip_time = value / self.clip.fps

    def _read_and_process_frame(self):
        rgb_image = self.clip.get_frame(self.clip_time)
        bgr_image = self._process_frame(rgb_image)
        cv2.imshow(self.win_name, bgr_image)

    def create_video(self, output_video_path='../output/project_video.mp4'):
        clip = VideoFileClip(self.video_path).fl_image(self._process_frame_rgb)
        clip.write_videofile(output_video_path, audio=False)

    def _process_frame(self, rgb_image):
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        draw_image = np.copy(bgr_image)
        self.draw.colorbar(draw_image, ticks=np.arange(self.confidence_range[0], self.confidence_range[1] + 1))

        classified_objects = []
        for search_window in self.search_windows.next():
            patch = cv2.resize(bgr_image[search_window.top:search_window.bottom,
                                         search_window.left:search_window.right], (64, 64))
            prediction, confidence = self.classifier.classify(patch)
            if prediction:
                classified_objects.append(ClassifiedObject(search_window=search_window, confidence=confidence))
                #  self.draw.box(draw_image, box=search_window, value=confidence)

        clustered_objects, heatmap = self.cluster.cluster(classified_objects)
        #cv2.imshow("Heatmap", cv2.applyColorMap((heatmap * 10).astype(np.uint8), cv2.COLORMAP_HOT))
        tracked_objects = self.tracker.track(clustered_objects)

        for obj in tracked_objects:
            self.draw.box(draw_image, box=obj.bbox, value=obj.confidence)
        return draw_image

    def _process_frame_rgb(self, rgb_image):
        """Process frame and convert output to RGB format"""
        bgr_image = self._process_frame(rgb_image)
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    VehicleDetectionPipeline().player()
    # VehicleDetectionPipeline().create_video()
