import cv2
from moviepy.editor import VideoFileClip
import sys
sys.path.append("../")
from src.lane_filter import LaneFilter


# constants
window_settings = {
    'INPUT_WIDTH': 640,
    'INPUT_HEIGHT': 360,
    'INPUT_POS_X': 0,
    'INPUT_POS_Y': 0,

    'OUTPUT_WIDTH': 640,
    'OUTPUT_HEIGHT': 360,
    'OUTPUT_POS_X': 0,
    'OUTPUT_POS_Y': 360,

    'FILTERS_GRID_ROWS': 3,
    'FILTERS_GRID_COLS': 3,
    'FILTERS_GRID_WIDTH': 640,
    'FILTERS_GRID_HEIGHT': 480,  # Extra space for sliders
    'FILTERS_GRID_POS_X': 640,  # Position of first filter window.
    'FILTERS_GRID_POS_Y': 0,  # Position of first filter window.
}


def update(lane_filter):
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    lane_filter.apply(gray_image)
    cv2.imshow(win_input_image, bgr_image)
    lane_filter.show()


def process_image(rgb_image):
    pause = False
    while 1:
        update(lane_filter)
        k = cv2.waitKey(20) & 0xff
        if k == ascii_dict['space']:
            pause = not pause  # toggle boolean
        if k == ascii_dict['esc']:
            return True  # quit
        if not pause:
            break
    return False


lane_filter = LaneFilter()
lane_filter.init_show(window_settings)

win_input_image = 'Input image'
cv2.namedWindow(win_input_image, cv2.WINDOW_NORMAL)
cv2.moveWindow(win_input_image, window_settings['INPUT_POS_X'], window_settings['INPUT_POS_Y'])
cv2.resizeWindow(win_input_image, window_settings['INPUT_WIDTH'], window_settings['INPUT_HEIGHT'])
clip = VideoFileClip('../input/project_video.mp4')

ascii_dict = {'esc': 27, 'space': 32}
for rgb_image in clip.iter_frames():
    quit = process_image(rgb_image)
    if quit:
        break
cv2.getWindowProperty
