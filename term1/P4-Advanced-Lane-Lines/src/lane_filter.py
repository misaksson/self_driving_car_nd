import numpy as np
import cv2
from functools import reduce


class ImageFilter(object):
    def __init__(self, thresh=(20, 100)):
        self.lower_th = thresh[0]
        self.upper_th = thresh[1]

    def apply(self, image):
        self.mask = (image >= self.lower_th) & (image <= self.upper_th)
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

    def apply(self, images):
        sobel = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        scaled = ((abs_sobel * 255.0) / np.max(abs_sobel)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelY(ImageFilter):
    def __init__(self, ksize=9, thresh=(20, 100)):
        self.ksize = ksize
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        sobel = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        scaled = ((abs_sobel * 255.0) / np.max(abs_sobel)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelMagnitude(ImageFilter):
    def __init__(self, ksize=9, thresh=(30, 100)):
        self.ksize = ksize
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        # ToDo: Cache sobel results (should be possible to reuse if same ksize).
        sobelx = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled = ((magnitude * 255.0) / np.max(magnitude)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelDirection(ImageFilter):
    def __init__(self, ksize=15, thresh=(0.7, 1.3)):
        self.ksize = ksize
        scaled_threshold = (np.uint8(thresh[0] * 255 / (np.pi / 2)),
                            np.uint8(thresh[1] * 255 / (np.pi / 2)))
        ImageFilter.__init__(self, thresh=scaled_threshold)

    def apply(self, images):
        sobelx = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        grad_direction = np.arctan2(np.abs(sobely), np.abs(sobelx))
        scaled = ((grad_direction * 255.0) / (np.pi / 2)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class Hue(ImageFilter):
    def __init__(self, thresh=(0, 255)):
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        return ImageFilter.apply(self, images['hls'][:, :, 0])


class Lightness(ImageFilter):
    def __init__(self, thresh=(0, 255)):
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        return ImageFilter.apply(self, images['hls'][:, :, 1])


class Saturation(ImageFilter):
    def __init__(self, thresh=(0, 255)):
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        return ImageFilter.apply(self, images['hls'][:, :, 2])


class Red(ImageFilter):
    def __init__(self, thresh=(0, 255)):
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        return ImageFilter.apply(self, images['rgb'][:, :, 0])


class Green(ImageFilter):
    def __init__(self, thresh=(0, 255)):
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        return ImageFilter.apply(self, images['rgb'][:, :, 0])


class Blue(ImageFilter):
    def __init__(self, thresh=(0, 255)):
        ImageFilter.__init__(self, thresh=thresh)

    def apply(self, images):
        return ImageFilter.apply(self, images['rgb'][:, :, 0])


class LaneFilter(ImageFilter):
    def __init__(self):
        self.image_filters = []
        self.image_filters.append(Hue())
        self.image_filters.append(Lightness())
        self.image_filters.append(Saturation())
        self.image_filters.append(Red())
        self.image_filters.append(Green())
        self.image_filters.append(Blue())
        self.image_filters.append(SobelX())
        self.image_filters.append(SobelY())
        self.image_filters.append(SobelMagnitude())
        self.image_filters.append(SobelDirection())

    def apply(self, images):
        masks = []
        for image_filter in self.image_filters:
            masks.append(image_filter.apply(images))

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
