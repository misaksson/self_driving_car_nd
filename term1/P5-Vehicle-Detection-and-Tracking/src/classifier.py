import os
import pickle
import numpy as np
import cv2
from collections import namedtuple

from classifier_cache import ClassifierCache
from trainer import Trainer
from feature_extractor import extract_features

classifier_path = "./classifier.p"

# Classified objects with probability above this threshold will be cached.
classifier_hard_threshold = 0.5
# Classified objects with probability above this threshold will be output to application.
classifier_soft_threshold = 0.75

ClassifiedObject = namedtuple('ClassifiedObject', ['search_window', 'probability'])


class Classifier(object):
    def __init__(self, grid_generator, force_train=False, use_cache=True):
        self.grid_generator = grid_generator
        self.use_cache = use_cache
        if self._classifier_available() and not force_train:
            self._load_classifier()
        else:
            self._train_classifier()
            self._store_classifier()

        if self.use_cache:
            self.cache = ClassifierCache(classifier_path=classifier_path)

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

    def _classify_patch(self, patch):
        features = extract_features(patch, **self.feature_extractor_args)
        X = features.reshape(1, -1).astype(np.float64)
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        probabilities = self.classifier.predict_proba(X)[0]
        prediction = probabilities[1] > classifier_hard_threshold
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
                    classified_objects.append(ClassifiedObject(search_window=search_window, probability=probability))

            if self.use_cache:
                self.cache.add(frame_idx, classified_objects)

        soft_filtered_objects = []
        for idx in range(len(classified_objects)):
            if classified_objects[idx].probability > classifier_soft_threshold:
                soft_filtered_objects.append(classified_objects[idx])
        return soft_filtered_objects
