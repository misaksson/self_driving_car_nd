import numpy as np
import cv2
import pickle
import os
import sys
from moviepy.editor import VideoFileClip

sys.path.append("../src/")
from extractor import Extractor


# constants

clip_path = '../input/project_video.mp4'
# clip_path = '../input/challenge_video.mp4'
# clip_path = '../input/harder_challenge_video.mp4'

thresholds_path = '../src/thresholds.p'

ascii_dict = {'esc': 27, 'space': 32}

window_settings = {
    'GRID_ROWS': 3,
    'GRID_COLS': 4,
    'GRID_WIDTH': 600,
    'GRID_HEIGHT': 400,
    'GRID_RECT_X_OFFSET': 3,
    'GRID_RECT_Y_OFFSET': 25,
    'GRID_POS_X': 0,
    'GRID_POS_Y': 25,
}


def load_camera_calibration():
    with open("./calibration.p", 'rb') as fid:
        return pickle.load(fid)


class WindowGrid(object):
    def __init__(self):
        self.grid_idx = 0

    def next(self):
        grid_col = self.grid_idx // window_settings['GRID_ROWS']
        grid_row = self.grid_idx % window_settings['GRID_ROWS']
        pos_x = (window_settings['GRID_POS_X'] +
                 grid_col * (window_settings['GRID_WIDTH'] + window_settings['GRID_RECT_X_OFFSET']))
        pos_y = (window_settings['GRID_POS_Y'] +
                 grid_row * (window_settings['GRID_HEIGHT'] + window_settings['GRID_RECT_Y_OFFSET']))
        self.grid_idx += 1
        return pos_x, pos_y, window_settings['GRID_WIDTH'], window_settings['GRID_HEIGHT']


class ExtractorPlayer(object):
    def __init__(self, file_path, calibration):
        self.calibration = calibration
        self.clip = VideoFileClip(file_path)
        self.n_frames = np.round(self.clip.fps * self.clip.duration).astype(np.int)
        self.clip_time = 0
        self.extractor = Extractor(thresh=self._load_thresholds(), th_change_callback=self.th_change_callback)
        self._create_display_windows()

    def _create_display_windows(self):
        window_grid = WindowGrid()
        pos_x, pos_y, width, height = window_grid.next()
        self.win_input_image = 'Input image'
        cv2.namedWindow(self.win_input_image, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.win_input_image, pos_x, pos_y)
        cv2.resizeWindow(self.win_input_image, width, height)
        cv2.createTrackbar('frame', self.win_input_image, 0, self.n_frames - 1, self.frame_slider_callback)
        self.extractor.init_show(window_grid)

    def _process_frame(self):
        rgb_image = self.clip.get_frame(self.clip_time)
        rgb_image = self.calibration.undistort(rgb_image)
        images = {
            'rgb': rgb_image,
            'gray': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY),
            'hls': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS),
            'bgr': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
        }
        self.extractor.apply(images)
        cv2.imshow(self.win_input_image, images['bgr'])
        self.extractor.show()

    def process(self):
        pause = False
        while 1:
            if not pause and (self.clip_time < self.clip.duration):
                self._process_frame()
                frame_idx = np.round(self.clip.fps * self.clip_time).astype(np.int)
                cv2.setTrackbarPos('frame', self.win_input_image, frame_idx)
                self.clip_time += 1.0 / self.clip.fps

            k = cv2.waitKey(20) & 0xff
            if k == ascii_dict['space']:
                pause = not pause  # toggle
            elif k == ascii_dict['esc']:
                break

        self._save_thresholds()
        cv2.destroyAllWindows()

    def frame_slider_callback(self, value):
        self.clip_time = value / self.clip.fps
        self._process_frame()

    def th_change_callback(self):
        self._process_frame()

    def _load_thresholds(self):
        if not os.path.isfile(thresholds_path):
            return dict()

        with open(thresholds_path, 'rb') as fid:
            thresholds = pickle.load(fid)
        print("Loaded thresholds:")
        print(thresholds)
        return thresholds

    def _save_thresholds(self):
        thresholds = self.extractor.get_thresholds()
        print("Saving thresholds:")
        print(thresholds)
        with open(thresholds_path, 'wb') as fid:
            pickle.dump(thresholds, fid)


if __name__ == '__main__':
    cam_calibration = load_camera_calibration()
    player = ExtractorPlayer(clip_path, cam_calibration)
    player.process()
