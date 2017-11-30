import numpy as np
from scipy.ndimage.measurements import label
from collections import namedtuple

BoundingBox = namedtuple('BoundingBox', ['top', 'left', 'bottom', 'right'])
ClusteredObject = namedtuple('ClusteredObject', ['bbox', 'confidence'])


class Cluster(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.heatmaps = []
        self.max_n_heatmaps = 5
        self.heatmap_threshold = 2.5

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

        return clustered_objects, filtered_heatmap

    def _create_frame_heatmap(self, classified_objects):
        """Create a heatmap for detections in this frame

        The classification confidence score of an object is added to all pixels
        in the search window where the object was found.
        """
        frame_heatmap = np.zeros((self.image_height, self.image_width))
        for search_window, confidence in classified_objects:
            frame_heatmap[search_window.top:search_window.bottom, search_window.left:search_window.right] += confidence

        self.heatmaps.append(frame_heatmap)

    def _accumulate_heatmaps(self):
        accumulated = np.zeros((self.image_height, self.image_width))
        for heatmap in self.heatmaps:
            accumulated += heatmap
        return accumulated

    def _apply_threshold(self, heatmap):
        heatmap[heatmap <= self.heatmap_threshold] = 0
        return heatmap

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
