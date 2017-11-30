import cv2
import numpy as np
from colormap import cmap_builder
from moviepy.editor import VideoFileClip

from grid_generator import GridGenerators
from classifier import Classifier
from draw import Draw


class VehicleDetectionPipeline(object):
    def __init__(self, video_path='../input/project_video.mp4'):
        self.video_path = video_path
        confidence_cmap = cmap_builder('yellow', 'lime (w3c)', 'cyan')
        self.confidence_range = np.array([0.0, 5.0])
        self.draw = Draw(cmap=confidence_cmap, value_range=self.confidence_range)
        self.search_windows = GridGenerators(VideoFileClip(self.video_path).get_frame(0))
        self.classifier = Classifier(force_train=False)

    def player(self):
        for rgb_image in VideoFileClip(self.video_path).iter_frames():
            bgr_image = self.process_frame(rgb_image)

            cv2.imshow("Vehicle Detection", bgr_image)
            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break  # Quit by pressing Esc
            if k == 32:
                cv2.waitKey()  # Pause by pressing space
        else:
            cv2.waitKey()

        cv2.destroyAllWindows()

    def create_video(self, output_video_path='../output/project_video.mp4'):
        clip = VideoFileClip(self.video_path).fl_image(self.process_frame_rgb)
        clip.write_videofile(output_video_path, audio=False)

    def process_frame(self, rgb_image):
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        draw_image = np.copy(bgr_image)
        self.draw.colorbar(draw_image, ticks=np.arange(self.confidence_range[0], self.confidence_range[1] + 1))

        for search_window in self.search_windows.next():
            patch = cv2.resize(bgr_image[search_window.top:search_window.bottom,
                                         search_window.left:search_window.right], (64, 64))
            prediction, confidence = self.classifier.classify(patch)
            if prediction:
                self.draw.box(draw_image, box=search_window, value=confidence)

        return draw_image

    def process_frame_rgb(self, rgb_image):
        """Process frame and convert output to RGB format"""
        bgr_image = self.process_frame(rgb_image)
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    VehicleDetectionPipeline().player()
    # VehicleDetectionPipeline().create_video()
