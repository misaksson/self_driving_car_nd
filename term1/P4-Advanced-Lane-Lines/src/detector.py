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
        self.line_coeffs_history = []

    def find(self, image):
        """Find line

        Search for the line if no previous detection is available, otherwise refine previous detection.

        Arguments:
            image - birds-eye-view perspective of a binary image with extracted pixels.
        """
        if not self.line_coeffs_history:
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
        self.line_coeffs_history.append(self.line_coeffs)

        # Calculate line coordinates for demo.
        fitted_line_y_coords = np.linspace(0, h - 1, h)
        fitted_line_x_coords = (self.line_coeffs[0] * fitted_line_y_coords**2 +
                                self.line_coeffs[1] * fitted_line_y_coords +
                                self.line_coeffs[2])
        Line.demo_image[fitted_line_y_coords.astype(np.int32),
                        np.round(fitted_line_x_coords).astype(np.int32), :] = np.array([0, 255, 255])

        # Keep line for real-world curvature estimation etc.
        self.line_x_coords = np.round(fitted_line_x_coords).astype(np.int32)
        self.line_y_coords = fitted_line_y_coords.astype(np.int32)

    def _refine(self, image):
        """Refine previous detection using this frame.

        Fit a polynomial to extracted pixels close to previously detected line.

        Arguments:
            image - birds-eye-view perspective of a binary image with extracted pixels.
        """
        h, w = image.shape

        # The minimum number of extracted pixels that must be found within the search boundary to consider the result
        # as a valid line detection.
        min_n_pixels_of_a_valid_detection = 300

        # Defines search area boundary, e.g. x-distance from previous line detection.
        boundary_margin = 100
        all_y_coords = np.linspace(0, h - 1, h)
        left_boundary = (self.line_coeffs[0] * all_y_coords**2 +
                         self.line_coeffs[1] * all_y_coords +
                         self.line_coeffs[2] - boundary_margin)
        right_boundary = left_boundary + (2 * boundary_margin)

        # Get x and y coordinates of pixels that previously been identified to possibly belong to lane lines.
        extracted_y_coords, extracted_x_coords = image.nonzero()

        # Identify extracted pixel coordinates within search boundary
        line_indices = np.where(np.logical_and(extracted_x_coords >= left_boundary[extracted_y_coords],
                                               extracted_x_coords <= right_boundary[extracted_y_coords]))[0]

        # Gather resulting coordinates.
        line_x_coords = extracted_x_coords[line_indices]
        line_y_coords = extracted_y_coords[line_indices]

        # Least squares polynomial fit.
        line_coeffs = np.polyfit(line_y_coords, line_x_coords, deg=2)

        if len(line_x_coords) >= min_n_pixels_of_a_valid_detection:
            # Add this detection to history and calculate average detection.
            self.line_coeffs = self._average_line_coeffs(line_coeffs)
        else:
            # Use average from previous detections.
            self.line_coeffs = self._average_line_coeffs()

        # The code below are only for demo visualization.
        # Create polygon of search boundary for drawing.
        boundary_x = np.clip(np.concatenate((left_boundary, np.flipud(right_boundary))), 0, w - 1)
        boundary_y = np.concatenate((all_y_coords, np.flipud(all_y_coords)))
        Line.demo_image[boundary_y.astype(np.int32), np.round(boundary_x).astype(np.int32), 1] = 255
#        pts = np.column_stack((np.round(boundary_x).astype(np.int32), boundary_y.astype(np.int32)))
#        cv2.fillPoly(Line.demo_image, [pts], (0, 255, 0))

        color = {Line.Location.LEFT: np.array([0, 0, 255]), Line.Location.RIGHT: np.array([255, 0, 0])}
        Line.demo_image[line_y_coords, line_x_coords, :] = color[self.location]

        # Draw fitted line coordinates in yellow.
        fitted_line_y_coords = np.linspace(0, h - 1, h)
        fitted_line_x_coords = (line_coeffs[0] * fitted_line_y_coords**2 +
                                line_coeffs[1] * fitted_line_y_coords +
                                line_coeffs[2])
        fitted_line_x_coords = np.clip(fitted_line_x_coords, 0, w - 1)
        Line.demo_image[fitted_line_y_coords.astype(np.int32),
                        np.round(fitted_line_x_coords).astype(np.int32), :] = np.array([0, 255, 255])

        # Draw fitted average line coordinates in cyan.
        fitted_avg_line_x_coords = (self.line_coeffs[0] * fitted_line_y_coords**2 +
                                    self.line_coeffs[1] * fitted_line_y_coords +
                                    self.line_coeffs[2])
        fitted_avg_line_x_coords = np.clip(fitted_avg_line_x_coords, 0, w - 1)
        Line.demo_image[fitted_line_y_coords.astype(np.int32),
                        np.round(fitted_avg_line_x_coords).astype(np.int32), :] = np.array([255, 255, 0])

        # Keep line for real-world curvature estimation etc.
        self.line_x_coords = np.round(fitted_avg_line_x_coords).astype(np.int32)
        self.line_y_coords = fitted_line_y_coords.astype(np.int32)

    def _average_line_coeffs(self, line_coeffs=None):
        """Calculate average line coefficients.

        Use detection history from a few frames to calculate average line coefficients.

        Arguments:
            line_coeffs - Line detection in this frame, or None when a valid detection is missing.

        Returns:
            Average line coefficients.
        """
        max_history_length = 10

        if line_coeffs is not None:
            self.line_coeffs_history.append(line_coeffs)

        # Calculate the average of each coefficient individually.
        avg_line_coeffs = np.average(self.line_coeffs_history, axis=0)

        # Limit history length by removing oldest detection. This is also done when detections are missing in current
        # frame, such that a new search eventually will be triggered when losing track of the line.
        if len(self.line_coeffs_history) > max_history_length or line_coeffs is None:
            self.line_coeffs_history.pop(0)

        return avg_line_coeffs

    def calc_real_world_curvature(self, x_meter_per_pixel, y_meter_per_pixel):
        """Calculate line curve radius in meters

        Use pixel resolution in meters to fit a second degree polynomial in real-world coordinates. The line radius is
        then calculated near to the vehicle using this formula:
        R = (1 + f'(y)^2)^(3/2) / abs(f"(y)), where a second degree polynomial f(y) = Ay^2 + By + C
        have f'(y)= 2Ay + B and f"(y) = 2A, resulting in R = ((1 + (2Ay + B))^2)^(3/2) / abs(2A)

        Arguments:
            x_meter_per_pixel - measured when finding perspective transform
            y_meter_per_pixel - measured when finding perspective transform

        Returns:
            Line curve radius in meters.
        """
        line_coeffs = np.polyfit(self.line_y_coords * y_meter_per_pixel, self.line_x_coords * x_meter_per_pixel, 2)
        A = line_coeffs[0]
        B = line_coeffs[1]

        # Evaluate curve radius at the bottom of warped image, e.g. just in front of the vehicle.
        y_eval = np.max(self.line_y_coords) * y_meter_per_pixel
        curve_radius = ((1 + (2 * A * y_eval + B)**2)**(3 / 2)) / np.absolute(2 * A)
        return curve_radius


class Detector(object):
    def __init__(self, perspective):
        self.perspective = perspective
        self.lines = [Line(Line.Location.LEFT),
                      Line(Line.Location.RIGHT)]

    def find(self, image):
        lines_curvature = []
        Line.init_frame(image)
        for line in self.lines:
            line.find(image)
            lines_curvature.append(line.calc_real_world_curvature(self.perspective['x_meter_per_pixel'],
                                                                  self.perspective['y_meter_per_pixel']))
        return lines_curvature

    def get_lane_boundary(self):
        boundary_x = np.concatenate((self.lines[0].line_x_coords, np.flipud(self.lines[1].line_x_coords)))
        boundary_y = np.concatenate((self.lines[0].line_y_coords, np.flipud(self.lines[1].line_y_coords)))
        return [np.column_stack((boundary_x, boundary_y))]
