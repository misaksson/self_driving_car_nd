import os
import pickle
import numpy as np
import cv2
from collections import namedtuple
from colormap import cmap_builder

from drawer import *
from classifier_cache import ClassifierCache
from trainer import Trainer
from feature_extractor import extract_features

classifier_path = "./classifier.p"

# Classified objects with probability above this threshold will be cached.
cache_threshold = 0.50
# Classified objects with probability above this threshold will be used by the clustering.
cluster_threshold = 0.60
# Classified objects with probability above this threshold will be used by the tracker.
tracking_threshold = 0.75


class ClassifiedObject(namedtuple('ClassifiedObject', ['bbox', 'probability', 'confidence'])):
    def __new__(cls, bbox, probability, confidence=None):
        if confidence is None:
            confidence = cls._probability_to_score(probability)
        return super(ClassifiedObject, cls).__new__(cls, bbox, probability, confidence)

    def _probability_to_score(probability, a=0.0009080398201937553, b=-0.0009080398201937553, k=10):
        """Maps probabilities exponentially to confidence score value

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


class Classifier(object):
    def __init__(self, grid_generator, force_train=False, use_cache=True, show_display=True):
        self.grid_generator = grid_generator
        self.use_cache = use_cache
        self.show_display = show_display
        if self._classifier_available() and not force_train:
            self._load_classifier()
        else:
            self._train_classifier()
            self._store_classifier()

        if self.use_cache:
            self.cache = ClassifierCache(classifier_path=classifier_path)

        self.low_threshold = cache_threshold
        self.medium_threshold = cluster_threshold
        self.high_threshold = tracking_threshold

        self.drawer = Drawer(bbox_settings=BBoxSettings(
                             color=DynamicColor(cmap=cmap_builder('yellow', 'lime (w3c)', 'cyan'),
                                                value_range=[0.5, 1.0],
                                                colorbar=Colorbar(ticks=np.array([0.5, 0.75, 1.0]),
                                                                  pos=np.array([0.03, 0.97]),
                                                                  size=np.array([0.3, 0.01])))),
                             inplace=False)

        if self.show_display:
            self._init_display()

    def _classifier_available(self):
        result = False
        if os.path.isfile(classifier_path):
            result = True
        return result

    def _train_classifier(self):
        print("Training the classifier (using default settings)")
        self.classifier, self.feature_scaler, self.feature_extractor_args, _ = Trainer().extract_features().train()

    def _load_classifier(self):
        print("Loading the classifier from", classifier_path)
        with open(classifier_path, 'rb') as fid:
            self.classifier, self.feature_scaler, self.feature_extractor_args = pickle.load(fid)

    def _store_classifier(self):
        with open(classifier_path, 'wb') as fid:
            pickle.dump((self.classifier, self.feature_scaler, self.feature_extractor_args), fid)

    def _init_display(self, height=670, width=1200, x=1205, y=0):
        self.win = "Classifier - probabilities"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, width, height)
        cv2.moveWindow(self.win, x, y)
        cv2.createTrackbar('cluster th', self.win, int(self.medium_threshold * 100), 100, self._set_medium_th_callback)
        cv2.createTrackbar('tracker th', self.win, int(self.high_threshold * 100), 100, self._set_high_th_callback)

    def _set_medium_th_callback(self, value):
        self.medium_threshold = value / 100

    def _set_high_th_callback(self, value):
        self.high_threshold = value / 100

    def _classify_patch(self, patch):
        features = extract_features(patch, **self.feature_extractor_args)
        X = features.reshape(1, -1).astype(np.float64)
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        probabilities = self.classifier.predict_proba(X)[0]
        prediction = probabilities[1] > self.low_threshold
        return prediction, probabilities[1]

    def classify(self, frame_idx, bgr_image):
        classified_objects = None
        if self.use_cache:
            # Try to load objects, returns None if there are no record for this frame,
            classified_objects = self.cache.get(frame_idx)

        if classified_objects is None:
            classified_objects = []
            for search_window in self.grid_generator.next():
                patch = cv2.resize(bgr_image[search_window.top:search_window.bottom,
                                             search_window.left:search_window.right], (64, 64))
                prediction, probability = self._classify_patch(patch)
                if prediction:
                    classified_objects.append(ClassifiedObject(bbox=search_window, probability=probability))

            if self.use_cache:
                self.cache.add(frame_idx, classified_objects)

        # Apply the actual probability threshold (now that objects are stored in cache).
        medium_confidence_objects = []
        high_confidence_objects = []
        for idx in range(len(classified_objects)):
            if classified_objects[idx].probability > self.medium_threshold:
                medium_confidence_objects.append(classified_objects[idx])
            if classified_objects[idx].probability > self.high_threshold:
                high_confidence_objects.append(classified_objects[idx])

        classification_image = self.drawer.draw(bgr_image, high_confidence_objects)

        if self.show_display:
            cv2.imshow(self.win, classification_image)

        return medium_confidence_objects, high_confidence_objects, classification_image
