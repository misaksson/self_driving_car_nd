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
classifier_cache_threshold = 0.5
# Classified objects with probability above this threshold will be output to application.
classifier_threshold = 0.80

ClassifiedObject = namedtuple('ClassifiedObject', ['bbox', 'probability'])


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

        self.probability_threshold = classifier_threshold

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
        self.drawer = Drawer(bbox_settings=BBoxSettings(
                             color=DynamicColor(cmap=cmap_builder('yellow', 'lime (w3c)', 'cyan'),
                                                value_range=[0.5, 1.0],
                                                colorbar=Colorbar(ticks=np.array([0.5, 0.75, 1.0]),
                                                                  pos=np.array([0.03, 0.97]),
                                                                  size=np.array([0.3, 0.01])))),
                             inplace=False)
        self.win = "Classifier - probabilities"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, width, height)
        cv2.moveWindow(self.win, x, y)
        cv2.createTrackbar('threshold', self.win, int(self.probability_threshold * 100), 100, self._trackbar_callback)

    def _update_display(self, image, objects):
        cv2.imshow(self.win, self.drawer.draw(image, objects))

    def _trackbar_callback(self, value):
        self.probability_threshold = value / 100

    def _classify_patch(self, patch):
        features = extract_features(patch, **self.feature_extractor_args)
        X = features.reshape(1, -1).astype(np.float64)
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        probabilities = self.classifier.predict_proba(X)[0]
        prediction = probabilities[1] > classifier_cache_threshold
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
        output_objects = []
        for idx in range(len(classified_objects)):
            if classified_objects[idx].probability > self.probability_threshold:
                output_objects.append(classified_objects[idx])
        if self.show_display:
            self._update_display(bgr_image, output_objects)

        return output_objects
