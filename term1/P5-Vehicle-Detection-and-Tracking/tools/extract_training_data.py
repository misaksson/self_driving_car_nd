"""Extracts training examples from labeled images

Sample additional training data from Udacity's annotated dataset:
https://github.com/udacity/self-driving-car/tree/master/annotations
"""

import os
import sys
import csv
from collections import namedtuple
import cv2
import numpy as np
from random import randint

sys.path.append("../src/")
from grid_generator import GridGenerators, CameraParams


# file_path = "../data_sets/object-detection-crowdai/"
file_path = "../data_sets/object-dataset/"
csv_path = os.path.join(file_path, 'labels.csv')
vehicles_output_path = "../training_data/vehicles/extracted"
occluded_vehicles_output_path = "../training_data/vehicles/extracted/occluded"
non_vehicles_output_path = "../training_data/non-vehicles/extracted"
label_paths = {'Foreground': vehicles_output_path, 'Background': non_vehicles_output_path}

with open(csv_path, 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter=' ')

    raw_labels = [label for label in csv_reader]

BoundingBox = namedtuple('BoundingBox', ['top', 'left', 'bottom', 'right'])


class Label(namedtuple('Label', ['class_type', 'bbox', 'occluded', 'attributes'])):
    def __new__(cls, class_type, bbox, occluded=False, attributes=None):
        return super(Label, cls).__new__(cls, class_type, bbox, occluded, attributes)


def label_incorrect(raw_label):
    return (int(raw_label['ymin']) == int(raw_label['ymax']) or
            int(raw_label['xmin']) == int(raw_label['xmax']))


frame_labels = dict()
for raw_label in raw_labels:
    if label_incorrect(raw_label):
        continue
    frame_name = raw_label['Frame']
    bbox = BoundingBox(top=int(raw_label['ymin']), left=int(raw_label['xmin']),
                       bottom=int(raw_label['ymax']), right=int(raw_label['xmax']))
    label = Label(class_type=raw_label['Label'], bbox=bbox, occluded=raw_label['occluded'] == '1',
                  attributes=raw_label['attributes'])
    if frame_name in frame_labels:
        frame_labels[frame_name].append(label)
    else:
        frame_labels[frame_name] = [label]

image_height, image_width, _ = cv2.imread(os.path.join(file_path, raw_labels[0]['Frame'])).shape
grids = GridGenerators(image_height=image_height, image_width=image_width,
                       camera_params=CameraParams(horizontal_fov=54.13, vertical_fov=42.01, z=1.1, pitch=1, yaw=0.2))

# Extract all search windows, but keep scales separate to be able to get a equal distribution.
grid_scales = []
for generator in grids.generators:
    grid_scales.append([search_window for search_window in generator.next()])

search_params = grids.get_params()

search_window_areas = [(height * width) for _, (height, width), _, _ in grids.get_params()]


def get_foreground_labels(label):
    """Find search windows at roughly same size and position as the label"""
    label_height = label.bbox.bottom - label.bbox.top + 1
    label_width = label.bbox.right - label.bbox.left + 1

    foreground_labels = []
    for search_window_area, grid_scale in zip(search_window_areas, grid_scales):
        if abs((label_height * label_width) - search_window_area) / search_window_area < 0.3:
            for search_window in grid_scale:
                if ((abs(label.bbox.left - search_window.left) < (0.30 * label_width) and
                     abs(label.bbox.right - search_window.right) < (0.30 * label_width) and
                     abs(label.bbox.top - search_window.top) < (0.30 * label_height) and
                     abs(label.bbox.bottom - search_window.bottom) < (0.30 * label_height))):
                    foreground_labels.append(Label(class_type='Foreground', bbox=search_window,
                                                   occluded=label.occluded, attributes=label.attributes))

    return foreground_labels


def add_foreground_labels(labels):
    labels_out = list(labels)
    for label in labels:
        if label.class_type in ['Car']:
            labels_out += get_foreground_labels(label)
    return labels_out


def get_background_label(labels):
    n_attempts = 100

    # Pedestrians, bikers and TrafficLights are considered as background. To avoid overlapping background examples,
    # already existing 'Background' labels are also considered to be foreground.
    forground_types = ['Car', 'Truck', 'Background']

    # Get a random grid scale.
    grid_scale = grid_scales[randint(0, len(grid_scales) - 1)]
    # Try multiple times to find a valid background window.
    for i in range(n_attempts):
        # Get a random search window from the grid scale.
        background = grid_scale[randint(0, len(grid_scale) - 1)]
        for label in labels:
            if ((label.class_type not in forground_types or
                 label.bbox.left > background.right or
                 label.bbox.right < background.left or
                 label.bbox.top > background.bottom or
                 label.bbox.bottom < background.top)):
                pass  # Still OK
            else:
                break  # Not OK
        else:
            # All labels OK with this background.
            break
    else:
        # No valid background window was found,
        return None
    return Label(class_type='Background', bbox=background)


def add_background_labels(labels):
    # Balance up the Car labels with background labels
    n_background = sum([label.class_type == 'Foreground' for label in labels])
    for i in range(n_background):
        background = get_background_label(labels)
        if background is not None:
            labels.append(background)
    return labels


color_dict = {'Car': (0, 255, 0),
              'Truck': (255, 0, 0),
              'Pedestrian': (0, 0, 255),
              'TrafficLight': (255, 100, 100),
              'Biker': (0, 255, 255),
              'Foreground': (255, 255, 100),
              'Background': (255, 255, 255)}

for frame_name, labels in frame_labels.items():
    labels = add_foreground_labels(labels)
    labels = add_background_labels(labels)

    bgr_image = cv2.imread(os.path.join(file_path, frame_name))
    for idx, label in enumerate(labels):
        if label.class_type in ['Foreground', 'Background']:
            patch = cv2.resize(bgr_image[label.bbox.top:label.bbox.bottom,
                                         label.bbox.left:label.bbox.right], (64, 64))
            dir_path = label_paths[label.class_type]
            if label.occluded:
                dir_path = os.path.join(dir_path, 'occluded')
            patch_path = os.path.join(dir_path, os.path.splitext(frame_name)[0] + "_" + str(idx) + ".png")
            cv2.imwrite(patch_path, patch)

    for idx, label in enumerate(labels):
        cv2.rectangle(bgr_image,
                      (label.bbox.left, label.bbox.top),
                      (label.bbox.right, label.bbox.bottom),
                      color_dict[label.class_type], 3)
        if label.occluded:
            cv2.rectangle(bgr_image,
                          (label.bbox.left, label.bbox.top),
                          (label.bbox.right, label.bbox.bottom),
                          (0, 0, 0), 1)

    for (search_roi, _, _, _) in search_params:
        cv2.rectangle(bgr_image,
                      (search_roi.left, search_roi.top),
                      (search_roi.right, search_roi.bottom),
                      (0, 0, 0), 2)

    cv2.imshow('Labeled image', bgr_image)

    k = cv2.waitKey(20) & 0xff
    if k == 32:  # Pause
        k = cv2.waitKey() & 0xff
    if k == 27:
        break  # Quit by pressing Esc


