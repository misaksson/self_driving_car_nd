import cv2
import numpy as np

class Draw(object):
    def __init__(self, cmap, value_range=[0.0, 1.0]):
        self.cmap = cmap
        self.value_range = value_range

    def _value2bgr(self, value):
        saturated_value = min(max(value, self.value_range[0]), self.value_range[1])
        r, g, b, _ = self.cmap((saturated_value - self.value_range[0]) /
                               (self.value_range[1] - self.value_range[0]))
        bgr_color = (b * 255, g * 255, r * 255)
        return bgr_color

    def colorbar(self, image, ticks=np.array([0, 1]), pos=np.array([0.03, 0.97]), size=np.array([0.3, 0.01])):
        image_size = np.array(list(image.shape)[:2])
        bar_top, bar_left = (pos * image_size).astype(np.int)
        bar_height, bar_width = (size * image_size).astype(np.int)
        n_colors = 201
        color_height = np.int(bar_height / n_colors)
        min_value = ticks[0]
        max_value = ticks[-1]
        step = (max_value - min_value) / n_colors
        eps = 0.0001
        for idx, value in enumerate(np.arange(max_value, min_value - eps, -step)):
            color_top = bar_top + color_height * idx
            color_bottom = color_top + color_height
            color_left = bar_left
            color_right = bar_left + bar_width
            color = self._value2bgr(value)
            cv2.rectangle(image, (color_left, color_top), (color_right, color_bottom), color, cv2.FILLED)
            tick = ticks[np.where(np.logical_and(ticks > (value - step/2), ticks <= (value + step/2)))]
            if len(tick) > 0:
                cv2.rectangle(image, (color_right + 2, color_top + 1), (color_right + 4, color_bottom + 1), (0, 0, 0), cv2.FILLED)
                cv2.rectangle(image, (color_right + 1, color_top), (color_right + 3, color_bottom), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, f"{tick[0]:.0f}", (color_right + 6, color_top + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"{tick[0]:.0f}", (color_right + 5, color_top + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


    def box(self, image, box, value=0.0, thick=3):
        color = self._value2bgr(value)
        cv2.rectangle(image, (box.left, box.top), (box.right, box.bottom), color, thick)
