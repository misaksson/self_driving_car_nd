import cv2
import numpy as np

from grid_generator import GridGenerators


def draw_box(image, search_window, color=(255, 0, 0), thick=3):
    cv2.rectangle(image, (search_window.left, search_window.top), (search_window.right, search_window.bottom),
                  color, thick)


bgr_image = cv2.imread("../images/bbox-example-image.jpg")

search_windows = GridGenerators(bgr_image)


for search_window, color, roi in search_windows.next():
    draw_box(bgr_image, search_window, color=color)
    draw_box(bgr_image, roi, color=color)

cv2.imshow("GridImage", bgr_image)

cv2.waitKey()
