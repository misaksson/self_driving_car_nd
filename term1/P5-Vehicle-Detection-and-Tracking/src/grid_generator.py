import numpy as np
from collections import namedtuple
import itertools

RegionOfInterest = namedtuple('RegionOfInterest', ['top', 'left', 'bottom', 'right'])
WindowSize = namedtuple('WindowSize', ['height', 'width'])
WindowOverlap = namedtuple('WindowOverlap', ['vertical', 'horizontal'])
SearchWindow = namedtuple('SearchWindow', ['top', 'left', 'bottom', 'right'])


class CameraParams(namedtuple('CameraParams', ['horizontal_fov', 'vertical_fov',  # degrees
                                               'x', 'y', 'z',  # longitudinal, lateral and vertical position in meters
                                               'roll', 'pitch', 'yaw'])):  # attitude in degrees
    def __new__(cls, horizontal_fov=None, vertical_fov=None, x=None, y=None, z=None, roll=None, pitch=None, yaw=None):
        return super(CameraParams, cls).__new__(cls, horizontal_fov, vertical_fov, x, y, z, roll, pitch, yaw)


class GridGenerator(object):
    colors = itertools.cycle([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)])

    def __init__(self, roi, window_size, window_overlap):
        self.roi = roi
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.color = next(GridGenerator.colors)

    def next(self):
        """Generates search windows from the grid"""
        start_x = self.roi.left
        end_x = self.roi.right - self.window_size.width + 1
        step_length_x = np.round(self.window_size.width * (1.0 - self.window_overlap.horizontal)).astype(np.int)
        start_y = self.roi.top
        end_y = self.roi.bottom - self.window_size.height + 1
        step_length_y = np.round(self.window_size.height * (1.0 - self.window_overlap.vertical)).astype(np.int)

        for top in range(start_y, end_y + 1, step_length_y):
            bottom = top + self.window_size.height - 1
            for left in range(start_x, end_x + 1, step_length_x):
                right = left + self.window_size.width - 1
                yield SearchWindow(top=top, left=left, bottom=bottom, right=right)

    def get_params(self):
        return (self.roi, self.window_size, self.window_overlap, self.color)


class GridGenerators(object):
    def __init__(self, image_height, image_width,
                 camera_params=CameraParams(horizontal_fov=54.13, vertical_fov=42.01, z=1.68, pitch=-3.02, yaw=0.2)):
        """Create grid generator for each scale

        A search corridor in front of the vehicle is defined, Multiple grid
        generators are then created to yield search windows at fixed distances
        in that corridor. The size of each grid and its search windows are
        calculated from an expected vehicle size and camera params that defines
        the perspective.

        Grid generators are ordered from just in front of the vehicle and
        onward, giving highest priority for objects near the vehicle in case
        there are an limited amount of processing time.

        Arguments:
            camera_params -- see named tuple definition above
        """
        degrees_per_x_pixel = camera_params.horizontal_fov / image_width
        degrees_per_y_pixel = camera_params.vertical_fov / image_height

        vanishing_point_x = (image_width / 2) + (camera_params.yaw / degrees_per_x_pixel)
        vanishing_point_y = (image_height / 2) - (camera_params.pitch / degrees_per_y_pixel)

        # Expected vehicle size. Note this does not need to be spot on to get detections although it might be good to
        # search for a few different sizes.
        vehicle_width_meters = 3.0
        vehicle_height_meters = 3.0

        # Defines a conic search corridor in front of vehicle, with grids placed at specific distances.
        roi_widths_meters = [28.0, 30.0, 38.0, 46.0, 54.0, 62.0, 70.0]
        roi_heights_meters = [7.375, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
        roi_distances_meters = [15.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]

        # Grid settings to use when doing a manual camera alignment calibration based on lanes width and dash length.
        # dash_length = 3.048
        # between_dash_length = 3 * dash_length
        # roi_distances_meters = [10, 10 + dash_length,  10 + dash_length + between_dash_length, 10 + dash_length * 2 + between_dash_length, 10 + dash_length * 2 + between_dash_length*2]

        self.generators = []

        # Calculate grid parameters for each search distance
        for roi_distance_meters, roi_width_meters, roi_heigth_meters in zip(roi_distances_meters,
                                                                            roi_widths_meters,
                                                                            roi_heights_meters):
            # roi_width_meters = 3.6576
            # roi_heigth_meters = 3.0

            degrees_per_meter = np.degrees(np.arctan(1.0 / roi_distance_meters))

            # Region of interest
            roi_width_pixels = roi_width_meters * degrees_per_meter / degrees_per_x_pixel
            roi_height_pixels = roi_heigth_meters * degrees_per_meter / degrees_per_y_pixel
            roi_left = (vanishing_point_x - roi_width_pixels / 2).astype(np.int)
            roi_right = np.minimum(image_width - 1, roi_left + roi_width_pixels).astype(np.int)
            roi_left = np.maximum(0, roi_left).astype(int)

            # Place 1/4 of the ROI below estimated road surface, skip this when calibrating
            vertical_offset = (1*roi_heigth_meters / 4)
            roi_bottom = vanishing_point_y + ((vertical_offset + camera_params.z) * degrees_per_meter /
                                              degrees_per_y_pixel)
            roi_top = np.maximum(0, roi_bottom - roi_height_pixels + 1).astype(np.int)
            roi_bottom = np.minimum(image_height - 1, roi_bottom).astype(np.int)

            # Search window size
            vehicle_width_pixels = (vehicle_width_meters * degrees_per_meter / degrees_per_x_pixel).astype(int)
            vehicle_height_pixels = (vehicle_height_meters * degrees_per_meter / degrees_per_y_pixel).astype(int)

            self.generators.append(GridGenerator(roi=RegionOfInterest(top=roi_top, left=roi_left,
                                                                      bottom=roi_bottom, right=roi_right),
                                                 window_size=WindowSize(height=vehicle_height_pixels,
                                                                        width=vehicle_width_pixels),
                                                 window_overlap=WindowOverlap(vertical=0.75, horizontal=0.75)))

    def next(self):
        for generator in self.generators:
            yield from generator.next()

    def get_params(self):
        grids_params = []
        for generator in self.generators:
            grid_params = generator.get_params()
            grids_params.append(grid_params)
        return grids_params
