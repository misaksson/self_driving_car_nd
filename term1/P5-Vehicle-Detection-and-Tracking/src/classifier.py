import os
import pickle
import numpy as np
import cv2
from collections import namedtuple

from trainer import Trainer
from feature_extractor import extract_features

classifier_path = "./classifier.p"

ClassifiedObject = namedtuple('ClassifiedObject', ['search_window', 'confidence'])


class Classifier(object):
    def __init__(self, grid_generator, force_train=False):
        self.grid_generator = grid_generator
        if self._classifier_available() and not force_train:
            self._load_classifier()
        else:
            self._train_classifier()
            self._store_classifier()

    def _classifier_available(self):
        result = False
        if os.path.isfile(classifier_path):
            result = True
        return result

    def _train_classifier(self):
        print("Training the classifier")
        self.classifier, self.feature_scaler, self.feature_extractor_args, _ = Trainer().train()

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
        scaled_X = self.feature_scaler.transform(X)
        confidence = self.classifier.decision_function(scaled_X)[0]
        prediction = confidence >= 0
        return prediction, confidence

    def classify(self, bgr_image):
        classified_objects = []
        for search_window in self.grid_generator.next():
            patch = cv2.resize(bgr_image[search_window.top:search_window.bottom,
                                         search_window.left:search_window.right], (64, 64))
            prediction, confidence = self._classify_patch(patch)
            if prediction:
                classified_objects.append(ClassifiedObject(search_window=search_window, confidence=confidence))
        return classified_objects
