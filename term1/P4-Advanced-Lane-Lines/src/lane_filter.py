import numpy as np
import cv2
from functools import reduce
from src.sobel_kernels import sobel_gain_factor


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
        self.show_result = True

    def apply(self, image):
        # There is some sporadic bug where the thresholds seems to be linked between instances.
        # I verified that the bug not is in the OpenCV trackbar callback mechanism by printing
        # self.name in the callback function, and only one instance is called. So there seems to
        # be some magic connection between ImageFilter instances, and as far as I noticed this
        # only happens when the thresholds are initialized to the exact same values in two or more
        # instances. Maybe python do some optimization by grouping what's considered to be const
        # values together between instances, not realizing that the thresholds are updated by the
        # OpenCV callback. The bug is present both in python 3.6.1 and 3.6.2. It does however
        # seem very unlikely that this is a python bug considering the number of users.
        #
        # This bug is easier to notice when the trackbars position are continuously updated to
        # actual (incorrect) value so lets do it here:
        cv2.setTrackbarPos("lower_th", self.name, self.thresh[0])
        cv2.setTrackbarPos("upper_th", self.name, self.thresh[1])

        self.mask = (image >= self.thresh[0]) & (image <= self.thresh[1])
        return self.mask

    def init_show(self, window_grid, thresh_slider=True):
        if not self.suppress_show:
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
        if not self.suppress_show:
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
        self.mask = reduce((lambda mask1, mask2: mask1 & mask2), masks)
        return self.mask


class MultiFilter_OR(MultiFilter):
    def combine(self, masks):
        self.mask = reduce((lambda mask1, mask2: mask1 | mask2), masks)
        return self.mask


class KernelFilters(ImageFilter):
    """Provides common initialization for kernel filters."""

    def __init__(self, ksize=5, **kwargs):
        self.ksize = ksize
        ImageFilter.__init__(self, **kwargs)


class SobelX(KernelFilters):
    def apply(self, images):
        sobel = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        scaled = (abs_sobel / sobel_gain_factor(self.ksize)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelY(KernelFilters):
    def apply(self, images):
        sobel = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        abs_sobel = np.abs(sobel)
        scaled = (abs_sobel / sobel_gain_factor(self.ksize)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelMagnitude(KernelFilters):
    def apply(self, images):
        # ToDo: Cache sobel results (should be possible to reuse if same ksize).
        sobelx = cv2.Sobel(images['gray'], cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(images['gray'], cv2.CV_64F, 0, 1, ksize=self.ksize)
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled = (magnitude / sobel_gain_factor(self.ksize)).astype(np.uint8)
        return ImageFilter.apply(self, scaled)


class SobelDirection(KernelFilters):
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


class YellowLineShadow(MultiFilter_AND):
    def __init__(self, **kwargs):
        MultiFilter.__init__(self, **kwargs)
        self.show_result = False
        self.append(SobelX)
        self.append(SobelY)


class YellowLineSunlight(MultiFilter_AND):
    def __init__(self, **kwargs):
        MultiFilter.__init__(self, **kwargs)
        self.show_result = False
        self.append(Hue)
        self.append(Saturation)
        self.append(Lightness)


class WhiteLine(MultiFilter_AND):
    def __init__(self, **kwargs):
        MultiFilter.__init__(self, **kwargs)
        self.show_result = False
        self.append(Red)


class Lines(MultiFilter_OR):
    def __init__(self, **kwargs):
        MultiFilter.__init__(self, **kwargs)
        self.show_result = False
        self.append(WhiteLine)
        self.append(YellowLineSunlight)
        self.append(YellowLineShadow)


class RoiMask(ImageFilter):
    """Region-of-interest mask

    Implemented as a ImageFilter, e.g. this is a hack using "thresholds" to
    select ROI size.
    """

    def apply(self, images):
        top = (255 - self.thresh[0]) / 255
        width = (255 - self.thresh[1]) / 255
        x1 = 0.5 - width
        x2 = 0.5 + width
        roi_vertices = np.array([[(0.1, 0.91),
                                  (x1, top),
                                  (x2, top),
                                  (0.9, 0.91)]],
                                dtype=np.float)
        roi_vertices = np.round(roi_vertices * images['gray'].shape[1::-1]).astype(int)
        mask = np.zeros_like(images['gray'])
        cv2.fillPoly(mask, roi_vertices, 1)
        self.mask = mask.astype(np.bool)
        return self.mask


class LaneFilter(MultiFilter_AND):
    def __init__(self, **kwargs):
        MultiFilter.__init__(self, **kwargs)
        self.append(Lines)
        self.append(RoiMask)
