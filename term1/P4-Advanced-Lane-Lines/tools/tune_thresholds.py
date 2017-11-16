import numpy as np
import cv2
import pickle
import os
import sys
import pandas as pd
from moviepy.editor import VideoFileClip

sys.path.append("../")
from src.lane_filter import LaneFilter


# constants
window_settings = {
    'INPUT_WIDTH': 640,
    'INPUT_HEIGHT': 420,
    'INPUT_POS_X': 0,
    'INPUT_POS_Y': 0,

    'OUTPUT_WIDTH': 640,
    'OUTPUT_HEIGHT': 360,
    'OUTPUT_POS_X': 0,
    'OUTPUT_POS_Y': 420,

    'FILTERS_GRID_ROWS': 3,
    'FILTERS_GRID_COLS': 3,
    'FILTERS_GRID_WIDTH': 640,
    'FILTERS_GRID_HEIGHT': 480,  # Extra space for sliders
    'FILTERS_GRID_POS_X': 640,  # Position of first filter window.
    'FILTERS_GRID_POS_Y': 0,  # Position of first filter window.
}
ascii_dict = {'esc': 27, 'space': 32}


def load_thresholds():
    if not os.path.isfile('./thresholds.p'):
        return dict()

    with open('./thresholds.p', 'rb') as fid:
        thresholds = pickle.load(fid)
    df = pd.DataFrame(thresholds)
    df.rename(index={0: 'Lower_th', 1: 'Upper_th'}, inplace=True)
    print(df)
    return thresholds


def save_thresholds(thresholds):
    with open('./thresholds.p', 'wb') as fid:
        pickle.dump(thresholds, fid)


def update():
    rgb_image = clip.get_frame(clip_time)
    images = {
        'rgb': rgb_image,
        'gray': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY),
        'hls': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS),
        'bgr': cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
    }
    lane_filter.apply(images)
    cv2.imshow(win_input_image, images['bgr'])
    lane_filter.show()


def process_image():
    pause = False
    while 1:
        update()
        k = cv2.waitKey(20) & 0xff
        if k == ascii_dict['space']:
            pause = not pause  # toggle boolean
        if k == ascii_dict['esc']:
            return True  # quit
        if not pause:
            break
    return False


def frame_slider_callback(value):
    global clip_time
    clip_time = value / clip.fps


lane_filter = LaneFilter(load_thresholds())
lane_filter.init_show(window_settings)

win_input_image = 'Input image'
cv2.namedWindow(win_input_image, cv2.WINDOW_NORMAL)
cv2.moveWindow(win_input_image, window_settings['INPUT_POS_X'], window_settings['INPUT_POS_Y'])
cv2.resizeWindow(win_input_image, window_settings['INPUT_WIDTH'], window_settings['INPUT_HEIGHT'])
clip = VideoFileClip('../input/project_video.mp4')
# clip = VideoFileClip('../input/challenge_video.mp4')
# clip = VideoFileClip('../input/harder_challenge_video.mp4')
n_frames = np.round(clip.fps * clip.duration).astype(np.int)
cv2.createTrackbar('frame', win_input_image, 0, n_frames - 1, frame_slider_callback)
clip_time = 0
while 1:
    quit = process_image()
    if clip_time < clip.duration:
        clip_time += 1.0 / clip.fps
    cv2.setTrackbarPos('frame', win_input_image, np.round(clip.fps * clip_time).astype(np.int))
    if quit:
        break

save_thresholds(lane_filter.get_thresholds())
cv2.destroyAllWindows()
