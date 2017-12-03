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

file_path = "../object-detection-crowdai/"
csv_path = os.path.join(file_path, 'labels.csv')
vehicles_output_path = "../training_data/vehicles/extracted"
non_vehicles_output_path = "../training_data/non-vehicles/extracted"
label_paths = {'Car': vehicles_output_path, 'Background': non_vehicles_output_path}

with open(csv_path, 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile)

    raw_labels = [label for label in csv_reader]


BoundingBox = namedtuple('BoundingBox', ['top', 'left', 'bottom', 'right'])
Label = namedtuple('Label', ['type', 'bbox'])


def label_invalid(raw_label):
    return (int(raw_label['ymin']) == int(raw_label['ymax']) or
            int(raw_label['xmin']) == int(raw_label['xmax']))


frame_labels = dict()
for raw_label in raw_labels:
    if label_invalid(raw_label):
        continue
    frame_name = raw_label['Frame']
    bbox = BoundingBox(top=int(raw_label['ymin']), left=int(raw_label['xmin']),
                       bottom=int(raw_label['ymax']), right=int(raw_label['xmax']))
    label = Label(type=raw_label['Label'], bbox=bbox)
    if frame_name in frame_labels:
        frame_labels[frame_name].append(label)
    else:
        frame_labels[frame_name] = [label]

image_height, image_width, _ = cv2.imread(os.path.join(file_path, raw_labels[0]['Frame'])).shape
grid = GridGenerators(image_height=image_height, image_width=image_width,
                      camera_params=CameraParams(horizontal_fov=54.13, vertical_fov=42.01, z=1.1, pitch=1, yaw=0.2))
search_windows = [search_window for search_window in grid.next()]
search_params = grid.get_params()


def get_background_label(labels):
    n_attempts = 100

    # Pedestrians and street lights are considered as background. To avoid overlapping background examples,
    # already existing 'Background' labels are also considered to be foreground.
    forground_types = ['Car', 'Truck', 'Background']

    # Try multiple times to find a valid background window.
    for i in range(n_attempts):
        # Get a random search window from the grid generator.
        background = search_windows[randint(0, len(search_windows) - 1)]
        for label in labels:
            if ((label.type not in forground_types or
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
    return Label(type='Background', bbox=background)


def add_background_labels(labels):
    # Balance up the Car labels with background labels
    n_background = sum([label.type == 'Car' for label in labels])
    for i in range(n_background):
        background = get_background_label(labels)
        if background is not None:
            labels.append(background)
    return labels


color_dict = {'Car': (0, 255, 0),
              'Truck': (255, 0, 0),
              'Pedestrian': (0, 0, 255),
              'Street Lights': (0, 255, 255),
              'Background': (255, 255, 255)}

for frame_name, labels in frame_labels.items():

    labels = add_background_labels(labels)

    bgr_image = cv2.imread(os.path.join(file_path, frame_name))
    for idx, label in enumerate(labels):
        if label.type in ['Car', 'Background']:
            patch = cv2.resize(bgr_image[label.bbox.top:label.bbox.bottom,
                                         label.bbox.left:label.bbox.right], (64, 64))
            cv2.imwrite(os.path.join(label_paths[label.type], os.path.splitext(frame_name)[0] + str(idx) + ".png"), patch)

    for idx, label in enumerate(labels):
        cv2.rectangle(bgr_image,
                      (label.bbox.left, label.bbox.top),
                      (label.bbox.right, label.bbox.bottom),
                      color_dict[label.type], 2)
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


