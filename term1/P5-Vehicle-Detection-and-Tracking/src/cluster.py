import numpy as np
import cv2
from scipy.ndimage.measurements import label
from collections import namedtuple
from colormap import cmap_builder

from drawer import *

BoundingBox = namedtuple('BoundingBox', ['top', 'left', 'bottom', 'right'])
ClusteredObject = namedtuple('ClusteredObject', ['bbox', 'confidence'])


class Cluster(object):
    def __init__(self, image_height, image_width, show_display=True):
        self.image_height = image_height
        self.image_width = image_width
        self.show_display = show_display
        self.heatmaps = []
        self.max_n_heatmaps = 5
        self.heatmap_threshold = 10
        if self.show_display:
            self._init_heatmap_display()

    def _init_heatmap_display(self, height=670, width=1200, x=1205, y=720):
        self.heat_drawer = Drawer(bbox_settings=BBoxSettings(
                                  color=DynamicColor(cmap=cmap_builder('black', 'red', 'yellow'),
                                                     value_range=[0, 255],
                                                     colorbar=Colorbar(ticks=np.array([0, 255]),
                                                                       pos=np.array([0.03, 0.96]),
                                                                       size=np.array([0.3, 0.01])))),
                                  inplace=False)
        self.cluster_drawer = Drawer(bbox_settings=BBoxSettings(
                                     color=DynamicColor(cmap=cmap_builder('yellow', 'lime (w3c)', 'cyan'),
                                                        value_range=[0, 20],
                                                        colorbar=Colorbar(ticks=np.array([0, 10, 20]),
                                                                          pos=np.array([0.03, 0.90]),
                                                                          size=np.array([0.3, 0.01])))),
                                     inplace=True)

        self.win = "Clustering - heatmap and confidence score"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, width, height)
        cv2.moveWindow(self.win, x, y)
        cv2.createTrackbar('threshold', self.win, self.heatmap_threshold, 255, self.set_heatmap_threshold)

    def set_heatmap_threshold(self, value):
        self.heatmap_threshold = value

    def _draw_heatmap(self, heatmap, objects):
        cmapped_image = self.heat_drawer.cmap(heatmap)
        cmapped_image = self.cluster_drawer.draw(cmapped_image, objects=objects)
        return cmapped_image

    def cluster(self, classified_objects):
        """Group classified objects into clusters

        Classified objects in current frame, aswell as the history is used to
        group/cluster multiple detections of the same object, and also remove
        detections of less influence that probably are false positives.

        Arguments:
            classified_objects -- list of objects from the classifier

        Returns:
            ClusteredObject list -- bounding box and confidence score
        """

        # Create a heatmap of the classified object in this frame.
        self._create_frame_heatmap(classified_objects)

        # Accumulate all heatmaps
        accumulated_heatmap = self._accumulate_heatmaps()

        if len(self.heatmaps) > self.max_n_heatmaps:
            # Drop the oldest heatmap
            self.heatmaps.pop(0)

        # Remove weak detections (false positives).
        filtered_heatmap = self._apply_threshold(accumulated_heatmap)

        # Find clustered objects in the heatmap
        clustered_objects = self._label(filtered_heatmap)

        # Create heatmap display image also showing clustered objects
        display_heatmap = self._draw_heatmap(accumulated_heatmap, clustered_objects)

        if self.show_display:
            cv2.imshow(self.win, display_heatmap)

        return clustered_objects, display_heatmap

    def _create_frame_heatmap(self, classified_objects):
        """Create a heatmap for detections in this frame

        The classification confidence score of an object is added to all pixels
        in the search window where the object was found.
        """
        frame_heatmap = np.zeros((self.image_height, self.image_width))
        for search_window, probability in classified_objects:
            frame_heatmap[search_window.top:search_window.bottom,
                          search_window.left:search_window.right] += self._probability_to_score(probability)

        self.heatmaps.append(frame_heatmap)

    def _probability_to_score(self, probability, a=0.0009080398201937553, b=-0.0009080398201937553, k=10):
        """Maps probabilities exponentially to cluster score value

        Like to have objects with very high probability to weight more than the
        combined output from several less confident classifications.

        The default arguments produce a curve that is quite flat near a score of
        0 until reaching an probability of 0.5-0.6 where the curvature starts,
        and from probability 0.8 to 1.0 the score is skyrocketing from 2.5 to
        20.

        Arguments:
            probability {[type]} -- [description]
        """
        score = a * np.exp(k * probability) + b
        return score

    def _accumulate_heatmaps(self):
        accumulated = np.zeros((self.image_height, self.image_width))
        for heatmap in self.heatmaps:
            accumulated += heatmap
        return accumulated

    def _apply_threshold(self, heatmap):
        filtered_heatmap = np.zeros((self.image_height, self.image_width))
        indicies_above_th = np.where(heatmap > self.heatmap_threshold)
        filtered_heatmap[indicies_above_th] = heatmap[indicies_above_th]
        return filtered_heatmap

    def _label(self, heatmap):
        """Find clustered objects in the heatmap

        The sklearn-label function assigns unique object ID's to nonzero
        groups/clusters of pixels in the heatmap. The ID is written to each
        pixel in of that cluster in the labeled image. Bounding boxes of a
        cluster can then be found by searching for the min- and max-coordinates
        having that ID.

        The output confidence score is calculated as the mean heatmap value in
        the bounding box. Note that since the heatmap is based on confidence
        scores from the classification, this does by some means just propagate
        the confidence score through the clustering step.
        """
        labeled_image, n_objects = label(heatmap)
        clustered_objects = []
        for object_id in range(1, n_objects + 1):
            y_coords, x_coords = np.where(labeled_image == object_id)
            bbox = BoundingBox(left=np.min(x_coords), right=np.max(x_coords),
                               top=np.min(y_coords), bottom=np.max(y_coords))
            confidence = heatmap[bbox.top:bbox.bottom, bbox.left:bbox.right].mean()
            clustered_objects.append(ClusteredObject(bbox=bbox, confidence=confidence))
        return clustered_objects
