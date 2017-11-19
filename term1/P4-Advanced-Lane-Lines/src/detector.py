import enum
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import cv2

cycol = cycle('rbgcmk')


class Line(object):
    class Location(enum.Enum):
        LEFT = 0,
        RIGHT = 1,

    @classmethod
    def init_frame(cls, image):
        cls.search_hist = None
        cls.demo_image = np.dstack((image, image, image)) * 255

    def __init__(self, location):
        self.location = location
        self.search_needed = True

    def find(self, image):
        """Find line

        Search for the line if no previous detection is available, otherwise refine previous detection.

        Arguments:
            image - birds-eye-view perspective of a binary image with extracted pixels.
        """
        if self.search_needed:
            self._search(image)
        else:
            self._refine(image)

    def _search(self, image):
        """Search for lane line position

        Find line

        Arguments:
            image - birds-eye-view perspective of a binary image with extracted pixels.
        """
        self._search_base(image)
        self._search_trace(image)

    def _search_base(self, image):
        """Search start of lane line.

        Find x position where the lane line intercepts the image bottom.

        Arguments:
            image - birds-eye-view perspective of a binary image with extracted pixels.
        """
        # Calculate a histogram as the vertical sum of lower half of the image.
        # This is only done once per frame.
        if Line.search_hist is None:
            Line.search_hist = np.sum(image[image.shape[0] // 2:, :], axis=0)
#            plt.figure()
#            plt.plot(Line.search_hist)
#            plt.scatter(line.base, 0, c=next(cycol), label=str(line.location))
#            plt.legend()
#            plt.show()

        midpoint = np.int(Line.search_hist.shape[0] / 2)
        if self.location == Line.Location.LEFT:
            self.base = np.argmax(Line.search_hist[:midpoint])
        elif self.location == Line.Location.RIGHT:
            self.base = np.argmax(Line.search_hist[midpoint:]) + midpoint

    def _search_trace(self, image):
        """Start from line base and trace line pixels.

        Use search windows to trace the line upwards in the binary image. This
        by sliding the search window for each iteration to the mean x-
        coordinate in previous search window.

        Arguments:
            image - birds-eye-view perspective of a binary image with extracted pixels.
        """
        h, w = image.shape

        # Number of vertical search windows to apply.
        n_search_windows = 9

        search_window_height = h // n_search_windows
        search_window_width = 200

        # Get x and y coordinates of pixels that previously been identified to possibly belong to lane lines.
        extracted_y_coords, extracted_x_coords = image.nonzero()

        # The minimum number of extracted pixels that must be found within a search window to consider it as a valid
        # line detection, such that the search window position can be updated to next frame.
        n_pixels_needed_to_recenter = 50

        line_indices = []

        # Start at bottom, at previously detected base coordinate and trace line upwards window for window.
        search_window_center_x = self.base
        for win_idx in range(n_search_windows):
            # Search boundary
            search_window_bottom = h - ((win_idx + 1) * search_window_height)
            search_window_top = h - (win_idx * search_window_height)
            search_window_left = search_window_center_x - (search_window_width // 2)
            search_window_right = search_window_center_x + (search_window_width // 2)

            # Draw search window.
            cv2.rectangle(Line.demo_image,
                          (search_window_left, search_window_bottom),
                          (search_window_right, search_window_top),
                          (0, 255, 0), 2)

            # Identify extracted pixel coordinates within current search window
            line_indices.append(np.where(np.logical_and(np.logical_and(extracted_y_coords >= search_window_bottom,
                                                                       extracted_y_coords < search_window_top),
                                                        np.logical_and(extracted_x_coords >= search_window_left,
                                                                       extracted_x_coords < search_window_right)))[0])
            # Slide window if a valid number of pixels was found in current window
            if len(line_indices[-1]) > n_pixels_needed_to_recenter:
                search_window_center_x = np.int(np.mean(extracted_x_coords[line_indices[-1]]))

        # Concatenate results from all search windows into one list of extracted coordinates indices.
        line_indices = np.concatenate(line_indices)

        # Gather resulting coordinates.
        line_x_coords = extracted_x_coords[line_indices]
        line_y_coords = extracted_y_coords[line_indices]

        color = {Line.Location.LEFT: np.array([0, 0, 255]), Line.Location.RIGHT: np.array([255, 0, 0])}
        Line.demo_image[line_y_coords, line_x_coords, :] = color[self.location]

        # Least squares polynomial fit.
        self.line_coeffs = np.polyfit(line_y_coords, line_x_coords, deg=2)

        # Calculate line coordinates for demo.
        fitted_line_y_coords = np.linspace(0, h - 1, h)
        fitted_line_x_coords = (self.line_coeffs[0] * fitted_line_y_coords**2 +
                                self.line_coeffs[1] * fitted_line_y_coords +
                                self.line_coeffs[2])
        Line.demo_image[fitted_line_y_coords.astype(np.int32),
                        np.round(fitted_line_x_coords).astype(np.int32), :] = np.array([0, 255, 255])
#        pts = np.column_stack((np.round(fitted_line_x_coords).astype(np.int32), fitted_line_y_coords.astype(np.int32)))
#        cv2.polylines(Line.demo_image, [pts], False, (0, 255, 255), 2)

        self.search_needed = False

    def _refine(self, image):
        """Refine previous detection using this frame.

        Fit a polynomial to extracted pixels close to previously detected line.

        Arguments:
            image - birds-eye-view perspective of a binary image with extracted pixels.
        """
        h, w = image.shape

        boundary_margin = 100
        all_y_coords = np.linspace(0, h - 1, h)
        left_boundary = (self.line_coeffs[0] * all_y_coords**2 +
                         self.line_coeffs[1] * all_y_coords +
                         self.line_coeffs[2] - boundary_margin)
        right_boundary = left_boundary + (2 * boundary_margin)

        # Get x and y coordinates of pixels that previously been identified to possibly belong to lane lines.
        extracted_y_coords, extracted_x_coords = image.nonzero()

        line_indices = np.where(np.logical_and(extracted_x_coords >= left_boundary[extracted_y_coords],
                                               extracted_x_coords <= right_boundary[extracted_y_coords]))[0]

        # Gather resulting coordinates.
        line_x_coords = extracted_x_coords[line_indices]
        line_y_coords = extracted_y_coords[line_indices]

        # Least squares polynomial fit.
        self.line_coeffs = np.polyfit(line_y_coords, line_x_coords, deg=2)

        boundary_x = np.concatenate((left_boundary, np.flipud(right_boundary)))
        boundary_y = np.concatenate((all_y_coords, np.flipud(all_y_coords)))
        Line.demo_image[boundary_y.astype(np.int32), np.round(boundary_x).astype(np.int32), 1] = 255

#        pts = np.column_stack((np.round(boundary_x).astype(np.int32), boundary_y.astype(np.int32)))
#        cv2.fillPoly(Line.demo_image, [pts], (0, 255, 0))

        color = {Line.Location.LEFT: np.array([0, 0, 255]), Line.Location.RIGHT: np.array([255, 0, 0])}
        Line.demo_image[line_y_coords, line_x_coords, :] = color[self.location]

        # Draw fitted line coordinates in yellow.
        fitted_line_y_coords = np.linspace(0, h - 1, h)
        fitted_line_x_coords = (self.line_coeffs[0] * fitted_line_y_coords**2 +
                                self.line_coeffs[1] * fitted_line_y_coords +
                                self.line_coeffs[2])
        Line.demo_image[fitted_line_y_coords.astype(np.int32),
                        np.round(fitted_line_x_coords).astype(np.int32), :] = np.array([0, 255, 255])


class Detector(object):
    def __init__(self):
        self.lines = [Line(Line.Location.LEFT),
                      Line(Line.Location.RIGHT)]

    def find(self, image):
        Line.init_frame(image)
        for line in self.lines:
            line.find(image)
            print(line.line_coeffs)
        cv2.imshow("Searched lines", Line.demo_image)
        Line.init_frame(image)
        for line in self.lines:
            line.find(image)
            print(line.line_coeffs)
        cv2.imshow("Refined lines", Line.demo_image)
