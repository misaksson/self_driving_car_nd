import numpy as np
import cv2
from functools import reduce


class ImageFilter(object):
    def __init__(self, thresh=(20, 100)):
        self.lower_th = thresh[0]
        self.upper_th = thresh[1]

    def apply(self, image):
        scaled = ((image * 255.0) / np.max(image)).astype(np.uint8)
        self.mask = (scaled > self.lower_th) & (scaled < self.upper_th)
        return self.mask

    def init_show(self, pos_x=0, pos_y=0, width=640, height=500, thresh_slider=True):
        cv2.namedWindow(self.__class__.__name__, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.__class__.__name__, pos_x, pos_y)
        cv2.resizeWindow(self.__class__.__name__, width, height)

        if thresh_slider:
            cv2.createTrackbar("lower_th", self.__class__.__name__, self.lower_th, 255,
                               self._adjust_lower_th)
            cv2.createTrackbar("upper_th", self.__class__.__name__, self.upper_th, 255,
                               self._adjust_upper_th)

    def show(self):
        cv2.imshow(self.__class__.__name__, self.mask.astype(np.uint8) * 255)

    def _adjust_lower_th(self, value):
        self.lower_th = value

    def _adjust_upper_th(self, value):
        self.upper_th = value


class SobelX(ImageFilter):
    def __init__(self, ksize=9, thresh=(20, 100)):
        self.ksize = ksize
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, image):
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        return ImageFilter.apply(self, abs_sobel)


class SobelY(ImageFilter):
    def __init__(self, ksize=9, thresh=(20, 100)):
        self.ksize = ksize
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, image):
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        return ImageFilter.apply(self, abs_sobel)


class LaneFilter(ImageFilter):
    def __init__(self):
        self.image_filters = []
        self.image_filters.append(SobelX())
        self.image_filters.append(SobelY())

    def apply(self, image):
        masks = []
        for image_filter in self.image_filters:
            masks.append(image_filter.apply(image))

        self.mask = reduce((lambda mask1, mask2: mask1 & mask2), masks).astype(np.uint8)
        return self.mask

    def init_show(self, window_settings):
        for idx, image_filter in enumerate(self.image_filters):
            grid_col = idx // window_settings['FILTERS_GRID_ROWS']
            grid_row = idx % window_settings['FILTERS_GRID_ROWS']
            pos_x = window_settings['FILTERS_GRID_POS_X'] + grid_col * window_settings['FILTERS_GRID_WIDTH']
            pos_y = window_settings['FILTERS_GRID_POS_Y'] + grid_row * window_settings['FILTERS_GRID_HEIGHT']
            image_filter.init_show(pos_x=pos_x, pos_y=pos_y,
                                   width=window_settings['FILTERS_GRID_WIDTH'],
                                   height=window_settings['FILTERS_GRID_HEIGHT'])
        ImageFilter.init_show(self,
                              pos_x=window_settings['OUTPUT_POS_X'],
                              pos_y=window_settings['OUTPUT_POS_Y'],
                              width=window_settings['OUTPUT_WIDTH'],
                              height=window_settings['OUTPUT_HEIGHT'],
                              thresh_slider=False)

    def show(self):
        for image_filter in self.image_filters:
            image_filter.show()
        ImageFilter.show(self)
