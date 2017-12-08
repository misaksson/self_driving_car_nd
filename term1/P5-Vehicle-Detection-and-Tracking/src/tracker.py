import numpy as np
from collections import namedtuple
from colormap import cmap_builder

from drawer import *

# Constants definition

# The tracked objects confidence is decreased by this factor each frame that lacks observation.
confidence_degradation_factor = 0.9

# The tracked objects confidence is increased by this factor each frame having an observation.
confidence_gradation_factor = 1.05

# Tracked objects with confidence below this threshold will be removed.
confidence_threshold = 12.0

# Tracked objects with confidence below this threshold will not be output.
output_confidence_threshold = 20.0

# Tracks are not included in output until reaching this age.
output_age_threshold = 5

# Two objects must get a match score below this value to be considered belonging to the same track.
match_score_threshold = 200


"""Internal tracked object states"""
TrackedObject = namedtuple('TrackedObject', ['age', 'confidence',
                                             'x', 'y', 'w', 'h', 'dx', 'dy', 'dw', 'dh'])

"""Bounding box used for output"""
BoundingBox = namedtuple('BoundingBox', ['top', 'left', 'bottom', 'right'])
"""Tracker object type used in output"""
TrackerOutputObject = namedtuple('TrackerOutputObject', ['bbox', 'confidence'])


class Tracker(object):
    def __init__(self, show_display=True):
        self.tracked_objects = []
        self.show_display = show_display
        self._init_params()
        if self.show_display:
            self._init_display()

    def _init_params(self):
        self.params = {'confidence_degradation_factor': confidence_degradation_factor,
                       'confidence_gradation_factor': confidence_gradation_factor,
                       'confidence_threshold': confidence_threshold,
                       'output_confidence_threshold': output_confidence_threshold,
                       'output_age_threshold': output_age_threshold,
                       'match_score_threshold': match_score_threshold,
                       }

    def _init_display(self, height=670, width=1200, x=0, y=720):
        self.drawer = Drawer(bbox_settings=BBoxSettings(
                             color=DynamicColor(cmap=cmap_builder('yellow', 'lime (w3c)', 'cyan'),
                                                value_range=[0, 100],
                                                colorbar=Colorbar(ticks=np.array([0, 50, 100]),
                                                                  pos=np.array([0.03, 0.96]),
                                                                  size=np.array([0.3, 0.01])))),
                             inplace=False)
        self.win = "Tracker - confidence score"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, width, height)
        cv2.moveWindow(self.win, x, y)
        cv2.createTrackbar('threshold', self.win, int(self.params['output_confidence_threshold']), 100,
                           self._trackbar_callback)

    def _trackbar_callback(self, value):
        self.params['output_confidence_threshold'] = float(value)

    def _update_display(self, image, objects):
        cv2.imshow(self.win, self.drawer.draw(image, objects))

    def track(self, bgr_image, clustered_objects):
        """Track objects in video images

        New observations (objects) are compared to track predictions, trying to
        find valid matches. Observations that match a track record are then
        influencing the new track estimate in proportion to the amount of
        confidence it brings to the track.

        When an observation not is considered to belong to any track, then it's
        used as a new track record. A track will however not be output until
        reaching a certain age.

        For tracks that do get any new observation, the state is updated to the
        prediction estimate for this frame. The confidence is then also
        decreased, which eventually will remove the track when there are no
        observations. There are two confidence thresholds, one for removing
        tracks and one for suppressing output of tracks, e.g. only keeping the
        track internally/hidden.

        Arguments:
            clustered_objects -- new observations

        Returns:
            TrackerOutputObjects -- tracked object estimates in this frame
        """
        self._calc_observations(clustered_objects)
        self._calc_predictions()
        self._update_tracks()
        self._remove_bad_tracks()
        output_objects = self._get_output_objects()
        if self.show_display:
            self._update_display(bgr_image, self._tracks_to_output_format(self.tracked_objects))
        return output_objects

    def _calc_observations(self, clustered_objects):
        """Convert clustered objects to track records"""
        self.observations = []
        for clustered_object in clustered_objects:
            width = (clustered_object.bbox.right - clustered_object.bbox.left + 1)
            height = (clustered_object.bbox.bottom - clustered_object.bbox.top + 1)
            self.observations.append(TrackedObject(age=0, confidence=clustered_object.confidence,
                                                   x=clustered_object.bbox.left + width / 2,
                                                   y=clustered_object.bbox.top + height / 2,
                                                   w=width, h=height, dx=0, dy=0, dw=0, dh=0))

    def _calc_predictions(self):
        """Predict the state of each track without considering new observations"""
        self.predicted_objects = []
        for track in self.tracked_objects:
            self.predicted_objects.append(TrackedObject(age=track.age + 1,
                                                        confidence=(track.confidence *
                                                                    self.params['confidence_degradation_factor']),
                                                        x=track.x + track.dx, y=track.y + track.dy,
                                                        w=track.w + track.dw, h=track.h + track.dh,
                                                        dx=track.dx, dy=track.dy, dw=track.dw, dh=track.dh))

    def _update_tracks(self):
        """Update tracks with new observations"""

        # Find objects that seem to match.
        obs2track_map, track2obs_map = self._match()
        for obs_idx, observed_object in enumerate(self.observations):
            if obs2track_map[obs_idx] is not None:
                self._add_observation_to_track(obs_idx, obs2track_map[obs_idx])
            else:
                # Add observation as a new track-record
                self.tracked_objects.append(observed_object)

        # Use prediction for tracks missing observation in this frame.
        for track_idx, obs_idx in enumerate(track2obs_map):
            if obs_idx is None:
                self.tracked_objects[track_idx] = self.predicted_objects[track_idx]

    def _add_observation_to_track(self, obs_idx, track_idx):
        """Add current observation into the track-record

        Update the track-record based on the new observation. The confidence
        of the observation and track-record are used to blend the both
        estimates.

        Arguments:
            obs_idx -- Index of observed object.
            track_idx -- Index of track-record and prediction.

        Returns:
            Observed to tracked object index map of matches.
            Indices of tracked objects lacking observation in this frame.
        """
        previous = self.tracked_objects[track_idx]
        prediction = self.predicted_objects[track_idx]
        observation = self.observations[obs_idx]

        # Blend track-record with the new observation based on the amount of confidence it brings.
        observation_portion = (observation.confidence / (observation.confidence + previous.confidence)) * 0.5
        track_portion = 1.0 - observation_portion

        updated_x = prediction.x * track_portion + observation.x * observation_portion
        updated_y = prediction.y * track_portion + observation.y * observation_portion
        updated_w = prediction.w * track_portion + observation.w * observation_portion
        updated_h = prediction.h * track_portion + observation.h * observation_portion

        track = TrackedObject(age=previous.age + 1,
                              confidence=((previous.confidence * track_portion +
                                           observation.confidence * observation_portion) *
                                          self.params['confidence_gradation_factor']),
                              x=updated_x, y=updated_y, w=updated_w, h=updated_h,
                              dx=updated_x - previous.x, dy=updated_y - previous.y,
                              dw=updated_w - previous.w, dh=updated_h - previous.h)

        self.tracked_objects[track_idx] = track

    def _match(self):
        """Find observations and tracks that match

        Calculates a match score between each observation and track prediction,
        which is building up a 2D match matrix. This matrix is then searched
        for the best match, which is added to output if it have a valid match
        score. Only one observation is allowed per track, and vise versa.
        Multiple matches is prevent by filling the column and row in the match
        matrix by invalid values.

        Returns:
            Index maps between detections, having None for missing match.
        """
        obs2track_map = [None for i in range(len(self.observations))]
        track2obs_map = [None for i in range(len(self.predicted_objects))]

        if len(self.observations) > 0 and len(self.predicted_objects) > 0:
            match_matrix = np.empty((len(self.observations), len(self.predicted_objects)))
            for obs_idx, observed_object in enumerate(self.observations):
                for pred_idx, predicted_object in enumerate(self.predicted_objects):
                    match_matrix[obs_idx, pred_idx] = self._calc_match_score(observed_object, predicted_object)

            while 1:
                obs_idx, pred_idx = np.unravel_index(match_matrix.argmin(), match_matrix.shape)
                if match_matrix[obs_idx, pred_idx] < self.params['match_score_threshold']:
                    obs2track_map[obs_idx] = pred_idx
                    track2obs_map[pred_idx] = obs_idx
                    match_matrix[obs_idx, :] = np.inf
                    match_matrix[:, pred_idx] = np.inf
                else:
                    # No more valid matches
                    break
        return obs2track_map, track2obs_map

    def _calc_match_score(self, object1, object2):
        """Calculate a match score for two objects

        Very basic comparison measure of two objects, the lower score the better match.
        """
        return (abs(object1.x - object2.x) + abs(object1.y - object2.y) +
                abs(object1.h - object2.h) + abs(object1.w - object2.w))

    def _remove_bad_tracks(self):
        """Remove tracks where the confidence is too low"""
        good_tracks = []
        for track in self.tracked_objects:
            if track.confidence > self.params['confidence_threshold']:
                good_tracks.append(track)
        self.tracked_objects = good_tracks

    def _tracks_to_output_format(self, tracks):
        output_objects = []
        for track in tracks:
            output_objects.append(TrackerOutputObject(bbox=BoundingBox(top=int(track.y - (track.h / 2)),
                                                                       left=int(track.x - (track.w / 2)),
                                                                       bottom=int(track.y + (track.h / 2)),
                                                                       right=int(track.x + (track.w / 2))),
                                                      confidence=track.confidence))
        return output_objects

    def _get_output_objects(self):
        """Provides valid output objects

        Convert objects having valid age and confidence to output format.

        Returns:
            TrackerOutputObjects -- Output objects from the tracker.
        """
        output_objects = []
        for track in self.tracked_objects:
            #print(track)
            if ((track.age >= self.params['output_age_threshold'] and
                 track.confidence >= self.params['output_confidence_threshold'])):

                output_objects.append(TrackerOutputObject(bbox=BoundingBox(top=int(track.y - (track.h / 2)),
                                                                           left=int(track.x - (track.w / 2)),
                                                                           bottom=int(track.y + (track.h / 2)),
                                                                           right=int(track.x + (track.w / 2))),
                                                          confidence=track.confidence))

        return output_objects
