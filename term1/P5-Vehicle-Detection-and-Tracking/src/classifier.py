import os
import pickle
import numpy as np

from trainer import Trainer
from feature_extractor import extract_features

classifier_path = "./classifier.p"


class Classifier(object):
    def __init__(self, force_train=False):
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
        self.classifier, self.feature_scaler, self.feature_extractor_args = Trainer().train()

    def _load_classifier(self):
        print("Loading the classifier from", classifier_path)
        with open(classifier_path, 'rb') as fid:
            self.classifier, self.feature_scaler, self.feature_extractor_args = pickle.load(fid)

    def _store_classifier(self):
        with open(classifier_path, 'wb') as fid:
            pickle.dump((self.classifier, self.feature_scaler, self.feature_extractor_args), fid)

    def classify(self, image):
        features = extract_features(image, **self.feature_extractor_args)
        X = features.reshape(1, -1).astype(np.float64)
        scaled_X = self.feature_scaler.transform(X)
        confidence = self.classifier.decision_function(scaled_X)[0]
        prediction = confidence >= 0
        return prediction, confidence
