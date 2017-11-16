import numpy as np
import cv2
from functools import reduce


class ImageFilter(object):
    """Base class for all filters.

    This provides basic thresholding and GUI windows to visualize the result
    and adjust threshold values using slider bars.
    """

    def __init__(self, thresh=[0, 255], parent_name=None, th_change_callback=None):
        self.thresh = thresh
        if parent_name is None:
            self.name = self.__class__.__name__
        else:
            self.name = parent_name + ":" + self.__class__.__name__
        self.th_change_callback = th_change_callback

    def apply(self, image):
        self.mask = (image >= self.thresh[0]) & (image <= self.thresh[1])
        return self.mask

    def init_show(self, window_grid, thresh_slider=True):

        pos_x, pos_y, width, height = window_grid.next()
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, pos_x, pos_y)
        cv2.resizeWindow(self.name, width, height)

        if thresh_slider:
            cv2.createTrackbar("lower_th", self.name, self.thresh[0], 255,
                               self._adjust_lower_th)
            cv2.createTrackbar("upper_th", self.name, self.thresh[1], 255,
                               self._adjust_upper_th)

    def show(self):
        cv2.imshow(self.name, self.mask.astype(np.uint8) * 255)

    def _adjust_lower_th(self, value):
        self.thresh[0] = value
        self._notify_listeners()

    def _adjust_upper_th(self, value):
        self.thresh[1] = value
        self._notify_listeners()

    def _notify_listeners(self):
        if self.th_change_callback is not None:
            self.th_change_callback()

    def get_thresholds(self):
        return self.thresh


class MultiFilter(ImageFilter):
    """Combines multiple ImageFilter's.

    This provides functionality to structure multiple ImageFilter's into
    hierarchies and combine their results as found suitable.

    Note that this class is abstract in that the combine method must be
    overridden.
    """

    def __init__(self, thresh=dict(), **kwargs):
        ImageFilter.__init__(self, thresh, **kwargs)
        self.image_filters = []

    def append(self, image_filter):
        if image_filter.__name__ in self.thresh:
            self.image_filters.append(image_filter(thresh=self.thresh[image_filter.__name__],
                                                   parent_name=self.name, th_change_callback=self.th_change_callback))
        else:
            print(f"Using default thresholds for {self.name}:{image_filter.__name__}")
            self.image_filters.append(image_filter(parent_name=self.name, th_change_callback=self.th_change_callback))

    def apply(self, images):
        masks = []
        for image_filter in self.image_filters:
            masks.append(image_filter.apply(images))

        self.mask = self.combine(masks)
        return self.mask

    def combine(masks):
        """Combine the results from multiple filter masks"""
        assert(False)  # This method must be overridden.

    def init_show(self, window_grid):
        for image_filter in self.image_filters:
            image_filter.init_show(window_grid)

        ImageFilter.init_show(self, window_grid, thresh_slider=False)

    def show(self):
        for image_filter in self.image_filters:
            image_filter.show()
        ImageFilter.show(self)

    def get_thresholds(self):
        thresh = dict()
        for image_filter in self.image_filters:
            thresh[image_filter.__class__.__name__] = image_filter.get_thresholds()

        return thresh


class MultiFilter_AND(MultiFilter):
    def combine(self, masks):
        self.mask = reduce((lambda mask1, mask2: mask1 & mask2), masks).astype(np.uint8)
        return self.mask


class MultiFilter_OR(MultiFilter):
    def combine(self, masks):
        self.mask = reduce((lambda mask1, mask2: mask1 | mask2), masks).astype(np.uint8)
        return self.mask


class KernelFilters(ImageFilter):
    """Provides common initialization for kernel filters."""

    def __init__(self, ksize=9, **kwargs):
        self.ksize = ksize
        ImageFilter.__init__(self, **kwargs)


class SobelX(KernelFilters):
    def apply(self, images):
        sobel = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        scaled = ((abs_sobel * 255.0) / np.max(abs_sobel)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelY(KernelFilters):
    def apply(self, images):
        sobel = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        scaled = ((abs_sobel * 255.0) / np.max(abs_sobel)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelMagnitude(KernelFilters):
    def apply(self, images):
        # ToDo: Cache sobel results (should be possible to reuse if same ksize).
        sobelx = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled = ((magnitude * 255.0) / np.max(magnitude)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelDirection(KernelFilters):
    def __init__(self, ksize=15, thresh=[0.7, 1.3], **kwargs):
        self.ksize = ksize
        scaled_threshold = (np.uint8(thresh[0] * 255 / (np.pi / 2)),
                            np.uint8(thresh[1] * 255 / (np.pi / 2)))
        ImageFilter.__init__(self, thresh=scaled_threshold, **kwargs)

    def apply(self, images):
        # ToDo: Cache sobel results (should be possible to reuse if same ksize).
        sobelx = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        grad_direction = np.arctan2(np.abs(sobely), np.abs(sobelx))
        scaled = ((grad_direction * 255.0) / (np.pi / 2)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class Hue(ImageFilter):
    def apply(self, images):
        return ImageFilter.apply(self, images['hls'][:, :, 0])


class Lightness(ImageFilter):
    def apply(self, images):
        return ImageFilter.apply(self, images['hls'][:, :, 1])


class Saturation(ImageFilter):
    def apply(self, images):
        return ImageFilter.apply(self, images['hls'][:, :, 2])


class Red(ImageFilter):
    def apply(self, images):
        return ImageFilter.apply(self, images['rgb'][:, :, 0])


class Green(ImageFilter):
    def apply(self, images):
        return ImageFilter.apply(self, images['rgb'][:, :, 1])


class Blue(ImageFilter):
    def apply(self, images):
        return ImageFilter.apply(self, images['rgb'][:, :, 2])


class YellowLine(MultiFilter_AND):
    def __init__(self, **kwargs):
        MultiFilter.__init__(self, **kwargs)
        self.append(Red)
        self.append(SobelY)


class LaneFilter(MultiFilter_OR):
    def __init__(self, **kwargs):
        MultiFilter.__init__(self, **kwargs)
        self.append(Hue)
        self.append(Red)
        self.append(YellowLine)
