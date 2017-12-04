import cv2
import numpy as np
from collections import namedtuple


DetectionObject = namedtuple('DetectionObject', ['bbox', 'value'])


class Colorbar(namedtuple('Colorbar', ['ticks', 'pos', 'size'])):
    def __new__(cls, ticks, pos=np.array([0.03, 0.97]), size=np.array([0.3, 0.01])):
        return super(Colorbar, cls).__new__(cls, ticks, pos, size)


class DynamicColor(namedtuple('DynamicColor', ['cmap', 'value_range', 'colorbar'])):
    def __new__(cls, cmap, value_range, colorbar=None):
        return super(DynamicColor, cls).__new__(cls, cmap, value_range, colorbar)

    def get_color(self, value):
        saturated_value = min(max(value, self.value_range[0]), self.value_range[1])
        r, g, b, _ = self.cmap((saturated_value - self.value_range[0]) /
                               (self.value_range[1] - self.value_range[0]))
        return (b * 255, g * 255, r * 255)


class StaticColor(namedtuple('StaticColor', ['color'])):
    colorbar = None

    def __new__(cls, color=(0, 0, 255)):
        return super(StaticColor, cls).__new__(cls, color)

    def get_color(self, value):
        return self.color


class BBoxSettings(namedtuple('BBoxSettings', ['color', 'border_thickness', 'alpha_fill'])):
    def __new__(cls, color=StaticColor(), border_thickness=3, alpha_fill=0.0):
        return super(BBoxSettings, cls).__new__(cls, color, border_thickness, alpha_fill)


class Drawer(object):
    def __init__(self, bbox_settings=BBoxSettings(), inplace=False):
        """Create a drawer annotating image with bounding boxes

        Keyword Arguments:
            bbox_settings {BBoxSettings} -- Defines how boxes are drawn (default: {BBoxSettings()})
            inplace {bool} -- Draw in input image if True, otherwise draw in a copy (default: {False})
        """
        self.bbox_settings = bbox_settings
        self.inplace = inplace

    def draw(self, image, objects):
        """Draw bounding boxes on image

        Arguments:
            image -- to be drawn on
            objects {[DetectionObject]} -- list of objects compatible with DetectionObject

        Returns:
            image -- the annotated image
        """
        if self.inplace:
            self.image = image
        else:
            self.image = np.copy(image)

        for bbox, value in objects:
            color = self.bbox_settings.color.get_color(value)
            self._draw_box(bbox, color)

        if self.bbox_settings.color.colorbar is not None:
            self._draw_colorbar()
        return self.image

    def cmap(self, image):
        color_mapped = (self.bbox_settings.color.cmap(image / 255) * 255).astype(np.uint8)
        self.image = cv2.cvtColor(color_mapped, cv2.COLOR_RGB2BGR)
        self._draw_colorbar()
        return self.image

    def _draw_colorbar(self):
        colorbar = self.bbox_settings.color.colorbar
        image_size = np.array(list(self.image.shape)[:2])
        bar_top, bar_left = (colorbar.pos * image_size).astype(np.int)
        bar_height, bar_width = (colorbar.size * image_size).astype(np.int)
        n_colors = 201
        color_height = np.int(bar_height / n_colors)
        min_value = colorbar.ticks[0]
        max_value = colorbar.ticks[-1]
        step = (max_value - min_value) / n_colors
        eps = 0.0001
        for idx, value in enumerate(np.arange(max_value, min_value - eps, -step)):
            color_top = bar_top + color_height * idx
            color_bottom = color_top + color_height
            color_left = bar_left
            color_right = bar_left + bar_width
            color = self.bbox_settings.color.get_color(value)
            cv2.rectangle(self.image, (color_left, color_top), (color_right, color_bottom), color, cv2.FILLED)

            # Draw tick if there is one at this position
            tick = colorbar.ticks[np.where(np.logical_and(colorbar.ticks > (value - step / 2),
                                                          colorbar.ticks <= (value + step / 2)))]
            if len(tick) > 0:
                cv2.rectangle(self.image, (color_right + 2, color_top + 1), (color_right + 4, color_bottom + 1),
                              (0, 0, 0), cv2.FILLED)
                cv2.rectangle(self.image, (color_right + 1, color_top), (color_right + 3, color_bottom),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(self.image, f"{tick[0]:.0f}", (color_right + 6, color_top + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.image, f"{tick[0]:.0f}", (color_right + 5, color_top + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_box(self, bbox, color):
        cv2.rectangle(self.image, (bbox.left, bbox.top), (bbox.right, bbox.bottom), color,
                      self.bbox_settings.border_thickness)
