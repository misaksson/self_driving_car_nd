import sys
import cv2
import numpy as np
import glob
import time
import pickle
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from feature_extractor import extract_features_from_files


class DummyScaler(object):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X


class Trainer(object):
    """Vehicle classifier trainer

    Loads image examples, extract features and train a classifier.
    """

    default_feature_extractor_args = {'color_space': 'YUV',
                                      'spatial_size': (32, 32),
                                      'hist_bins': 32,
                                      'orient': 12,
                                      'pix_per_cell': 8,
                                      'cell_per_block': 2,
                                      'hog_channels': [0, 1, 2],
                                      'spatial_feat': False,
                                      'hist_feat': True,
                                      'hog_feat': True,
                                      }

    default_classifier = RandomForestClassifier
    default_classifier_args = {'min_samples_split': 0.00237,
                               'max_features': 'sqrt',
                               'n_estimators': 10,
                               'criterion': "gini",
                               'max_depth': 100,
                               }

    def __init__(self, file_sets=[0, 1, 2]):
        """Classifier trainer

        Select files, extract features, train and validates a classifier.

        Arguments:
        file_sets {list} - selects which files to use for training
        """
        self._find_image_files(file_sets)
        self._balance_image_files()
        self._time_series_drop()
        self._train_test_split()

    def _find_image_files(self, file_sets):
        """Locates training example files

        Recursive search of image examples in the training_data folder.

        Arguments:
        file_sets {list} - selects which files to use for training
        """
        all_image_files = glob.glob("../training_data/**/*.png", recursive=True)

        # Group image files, separating foreground and background, resp. singles and series examples.

        self.file_groups = {'foreground_singles': [], 'foreground_series': [],
                            'background_singles': [], 'background_series': []}

        for image_file in all_image_files:
            if "non-vehicles" in image_file:
                if "Series/1" in image_file:
                    if 1 in file_sets:
                        self.file_groups['background_series'].append(image_file)
                elif "Series/2" in image_file:
                    if 2 in file_sets:
                        self.file_groups['background_series'].append(image_file)
                elif 0 in file_sets:
                    self.file_groups['background_singles'].append(image_file)
            elif "vehicles" in image_file:
                if "Series/1" in image_file:
                    if 1 in file_sets:
                        self.file_groups['foreground_series'].append(image_file)
                elif "Series/2" in image_file:
                    if 2 in file_sets and "occluded" not in image_file:
                        self.file_groups['foreground_series'].append(image_file)
                elif 0 in file_sets:
                    self.file_groups['foreground_singles'].append(image_file)
            else:
                assert(False)

        self.file_groups['foreground_singles'] = np.array(self.file_groups['foreground_singles'])
        self.file_groups['foreground_series'] = np.array(self.file_groups['foreground_series'])
        self.file_groups['background_singles'] = np.array(self.file_groups['background_singles'])
        self.file_groups['background_series'] = np.array(self.file_groups['background_series'])

        print("Background files:", len(self.file_groups['background_singles']), "singles +",
              len(self.file_groups['background_series']), "series")
        print("Foreground files:", len(self.file_groups['foreground_singles']), "singles +",
              len(self.file_groups['foreground_series']), "series")

    def _balance_image_files(self):
        """Equalize foreground and background samples

        Undersample the time-series data to equalize number of foreground and
        background samples. The singles samples are almost equal already.
        """
        if len(self.file_groups['background_series']) < len(self.file_groups['foreground_series']):
            key = 'foreground_series'
            keep_rate = len(self.file_groups['background_series']) / len(self.file_groups['foreground_series'])

        elif len(self.file_groups['background_series']) > len(self.file_groups['foreground_series']):
            key = 'background_series'
            keep_rate = len(self.file_groups['foreground_series']) / len(self.file_groups['background_series'])
        else:
            # Already equal
            return

        n_samples = len(self.file_groups[key])
        n_samples_to_keep = int(n_samples * keep_rate)
        indices = np.array(random.sample(range(n_samples), n_samples_to_keep))
        self.file_groups[key] = self.file_groups[key][indices]

        print("Background files after balancing:", len(self.file_groups['background_singles']), "singles +",
              len(self.file_groups['background_series']), "series")
        print("Foreground files after balancing:", len(self.file_groups['foreground_singles']), "singles +",
              len(self.file_groups['foreground_series']), "series")

    def _time_series_drop(self):
        """ Drop some random samples from time-series

        Used to drop a portion of the somewhat redundant samples in the time-series training sets. This to reduce
        training time, memory usage, and to avoid overfitting the data in the time-series.
        """
        time_series_drop_rate = 0.0  # Currently all data is used.

        n_background_series = len(self.file_groups['background_series'])
        n_foreground_series = len(self.file_groups['foreground_series'])

        if n_background_series > 0:
            n_background_to_keep = int(n_background_series * (1 - time_series_drop_rate))
            n_foreground_to_keep = int(n_foreground_series * (1 - time_series_drop_rate))

            background_indices = np.array(random.sample(range(n_background_series), n_background_to_keep))
            foreground_indices = np.array(random.sample(range(n_foreground_series), n_foreground_to_keep))

            self.file_groups['background_series'] = self.file_groups['background_series'][background_indices]
            self.file_groups['foreground_series'] = self.file_groups['foreground_series'][foreground_indices]

        print("Background files after dropping:", len(self.file_groups['background_singles']), "singles +",
              len(self.file_groups['background_series']), "series")
        print("Foreground files after dropping:", len(self.file_groups['foreground_singles']), "singles +",
              len(self.file_groups['foreground_series']), "series")

    def _train_test_split(self):
        """Split example files into a train and test set

        Files are separated differently if they are singles or time-series
        examples. The singles are just shuffled and divided into sets. If doing
        the same for the time-series examples, there would be many examples in
        the test-set that look roughly the same as examples in the training-set,
        which might give good result in test set even if the training has
        overfitted the training set. In an attempt to avoid this, the
        time-series files are split before shuffling.
        """
        test_size = 0.2

        y_singles = np.concatenate((np.zeros(len(self.file_groups['background_singles'])),
                                    np.ones(len(self.file_groups['foreground_singles']))), axis=0)
        X_singles = np.concatenate((self.file_groups['background_singles'],
                                    self.file_groups['foreground_singles']), axis=0)
        X_singles_train, X_singles_test, y_singles_train, y_singles_test = train_test_split(
            X_singles, y_singles, test_size=test_size, random_state=42)

        n_background_series_test = int(len(self.file_groups['background_series']) * test_size)
        n_background_series_train = len(self.file_groups['background_series']) - n_background_series_test
        n_foreground_series_test = int(len(self.file_groups['foreground_series']) * test_size)
        n_foreground_series_train = len(self.file_groups['foreground_series']) - n_foreground_series_test

        X_series_test = np.concatenate((self.file_groups['background_series'][:n_background_series_test],
                                        self.file_groups['foreground_series'][:n_foreground_series_test]), axis=0)
        X_series_train = np.concatenate((self.file_groups['background_series'][n_background_series_test:],
                                         self.file_groups['foreground_series'][n_foreground_series_test:]), axis=0)
        y_series_test = np.concatenate((np.zeros(n_background_series_test),
                                        np.ones(n_foreground_series_test)), axis=0)
        y_series_train = np.concatenate((np.zeros(n_background_series_train),
                                         np.ones(n_foreground_series_train)), axis=0)

        X_series_test, y_series_test = shuffle(X_series_test, y_series_test, random_state=42)
        X_series_train, y_series_train = shuffle(X_series_train, y_series_train, random_state=42)

        # Concatenate the single and series data and shuffle.
        self.X_test_files, self.y_test = shuffle(np.concatenate((X_singles_test, X_series_test), axis=0),
                                                 np.concatenate((y_singles_test, y_series_test), axis=0),
                                                 random_state=42)
        self.X_train_files, self.y_train = shuffle(np.concatenate((X_singles_train, X_series_train), axis=0),
                                                   np.concatenate((y_singles_train, y_series_train), axis=0),
                                                   random_state=42)

    def extract_features(self, feature_extractor_args=None, X_scaler=None):
        """Extract features for training

        This must be called before the train method. Extraction is implemented
        separately from the training to be able to tune training without having
        to extract features.

        Keyword Arguments:
            feature_extractor_args {dict} -- see Trainer.default_feature_extractor_args
            X_scaler -- such as StandardScaler. The default (None) is to do no scaling.
        """
        if feature_extractor_args is not None:
            self.feature_extractor_args = feature_extractor_args
        else:
            self.feature_extractor_args = Trainer.default_feature_extractor_args

        X_train_features = extract_features_from_files(self.X_train_files, **self.feature_extractor_args)
        print("Extracted features from ", len(self.X_train_files),
              "train files, utilizing", sys.getsizeof(X_train_features), "bytes")

        X_test_features = extract_features_from_files(self.X_test_files, **self.feature_extractor_args)
        print("Extracted features from ", len(self.X_test_files),
              "test files, utilizing", sys.getsizeof(X_test_features), "bytes")

        if X_scaler is None:
            # No feature scaling (which not is necessary e.g. for decision trees)
            self.X_scaler = None
            self.X_test = X_test_features
            self.X_train = X_train_features
        else:
            # Standardize feature vector, fit the scaler using both train and test set
            X_temp = np.vstack((X_test_features, X_train_features)).astype(np.float64)
            self.X_scaler = X_scaler.fit(X_temp)
            self.X_test = self.X_scaler.transform(X_test_features)
            self.X_train = self.X_scaler.transform(X_train_features)

    def train(self, classifier=None):
        """Train and validate a classifier

        Note that the classifier should already be initialized. This is to make
        it easy to run warm_start training on another file-sets, which is easily
        done by just initializing another Trainer that uses a different file-set.
        The warm-start training can then continue at another point in time or why
        not on another machine, just by storing the intermediate result.

        Arguments:
            classifier -- an initialized classifier implementing the sklearn API (fit, score).

        Returns:
            classifier - the trained classifier
            X_scaler - the fitted feature scaler
            feature_extractor_args - provided for convenience
            test_accuracy - The validation score for the classifier.
        """
        if classifier is None:
            print(f"Training default classifier {Trainer.default_classifier.__name__}")
            self.classifier = Trainer.default_classifier(**Trainer.default_classifier_args)
        else:
            print("Using provided classifier instance.")
            self.classifier = classifier
        t1 = time.time()
        self.classifier.fit(self.X_train, self.y_train)
        print(f"{time.time() - t1:.1f} seconds to train")

        # Validate
        test_accuracy = self.classifier.score(self.X_test, self.y_test)
        print("Test Accuracy of classifier = ", round(test_accuracy, 4))

        return self.classifier, self.X_scaler, self.feature_extractor_args, test_accuracy

    def demo_predict(self, n_examples=10):
        t1 = time.time()
        indices = np.random.randint(len(self.X_test), size=n_examples)
        print(indices)
        print("Predicted: ", self.classifier.predict(self.X_test[indices]))
        print("Expected:", n_examples, "labels: ", self.y_test[indices])
        print(f"{time.time() - t1:.1f} seconds to predict {n_examples} examples")

    def show_false_positives(self):
        background_files = self.X_test_files[np.where(self.y_test == 0)]
        background_features = extract_features_from_files(background_files, **self.feature_extractor_args)
        X = np.array(background_features, dtype=np.float64)
        if self.X_scaler is not None:
            X = self.X_scaler.transform(X)
        probabilities = self.classifier.predict_proba(X)[:, 1]
        bad_indices = np.where(probabilities >= 0.5)
        files = np.array(background_files)[bad_indices]
        print("Num FP", len(files), "out of", len(background_files), "background files")
        print(f"Probabilities min: {np.min(probabilities):.4f}, max: {np.max(probabilities):.4f}, "
              f"avg: {np.mean(probabilities):.4f}")

        for file, probability in zip(files, probabilities[bad_indices]):
            print(f"The probability of {file} is {probability:.04f}")
            cv2.imshow("False Positive", cv2.imread(file))
            k = cv2.waitKey() & 0xff
            if k == 27:  # esc
                break
        cv2.destroyAllWindows()

    def show_false_negatives(self):
        foreground_files = self.X_test_files[np.where(self.y_test == 1)]
        foreground_features = extract_features_from_files(foreground_files, **self.feature_extractor_args)
        X = np.array(foreground_features, dtype=np.float64)
        if self.X_scaler is not None:
            X = self.X_scaler.transform(X)
        probabilities = self.classifier.predict_proba(X)[:, 1]
        bad_indices = np.where(probabilities < 0.5)
        files = np.array(foreground_files)[bad_indices]
        print("Num FN", len(files), "out of", len(foreground_files), "vehicle files")
        print(f"Probabilities min: {np.min(probabilities):.4f}, max: {np.max(probabilities):.4f}, "
              f"avg: {np.mean(probabilities):.4f}")

        for file, probability in zip(files, probabilities[bad_indices]):
            print(f"The probability of {file} is {probability:.04f}")
            cv2.imshow("False Negatives", cv2.imread(file))
            k = cv2.waitKey() & 0xff
            if k == 27:  # esc
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    trainer = Trainer(file_sets=[0])
    trainer.extract_features()
    trainer.train()
    trainer.demo_predict()
    trainer.show_false_negatives()
    trainer.show_false_positives()
