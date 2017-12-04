import sys
import cv2
import numpy as np
import glob
import time
import pickle
import random
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from feature_extractor import extract_features_from_files


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

    def __init__(self):
        self._find_image_files()
        self._balance_image_files()
        self._time_series_drop()
        self._train_test_split()

    def _find_image_files(self):
        """Locates training example files

        Recursive search of image examples in the training_data folder.
        """
        all_image_files = glob.glob("../training_data/**/*.png", recursive=True)

        # Group image files, separating foreground and background, resp. singles and series examples.

        self.file_groups = {'foreground_singles': [], 'foreground_series': [],
                            'background_singles': [], 'background_series': []}

        for image_file in all_image_files:
            if "non-vehicles" in image_file:
                if "Series" in image_file:
                    self.file_groups['background_series'].append(image_file)
                else:
                    self.file_groups['background_singles'].append(image_file)
            elif "vehicles" in image_file:
                if "Series" in image_file:
                    self.file_groups['foreground_series'].append(image_file)
                else:
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

        Ended up using too much data, causing problems with memory usage and training times. Since the correlation
        among samples in the time-series are high, I decided to drop some random samples.
        """
        time_series_drop_rate = 0.9

        n_background_series = len(self.file_groups['background_series'])
        n_foreground_series = len(self.file_groups['foreground_series'])

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

        Files are separated differently if they are singles or series examples.
        The singles examples are just shuffled and divided into sets. If doing
        the same for the series examples, there would be many examples in the
        test set that look roughly the same as examples in the training set,
        which might give good result in test set even if the training has
        overfitted the training set. In an attempt to avoid this,
        the series files are split before shuffling.
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

    def extract_features(self, feature_extractor_args=None):
        """Extract features for training

        This must be called before the train method. Extraction is implemented
        separately from the training to be able to tune training without having
        to extract features.

        Keyword Arguments:
            feature_extractor_args {dict} -- see Trainer.default_feature_extractor_args
        """
        if feature_extractor_args is not None:
            self.feature_extractor_args = feature_extractor_args
        else:
            self.feature_extractor_args = Trainer.default_feature_extractor_args

        X_test_features = extract_features_from_files(self.X_test_files, **self.feature_extractor_args)
        X_train_features = extract_features_from_files(self.X_train_files, **self.feature_extractor_args)

        print("Extracted features from ", len(self.X_test_files),
              "test files, utilizing", sys.getsizeof(X_test_features), "bytes")
        print("Extracted features from ", len(self.X_train_files),
              "train files, utilizing", sys.getsizeof(X_train_features), "bytes")

#        # Dumping features to file to free up some memory before fitting StandardScaler
#        dump_file_path = f"../output/features.p"
#        with open(dump_file_path, "wb") as fid:
#            print("Writing features to", dump_file_path)
#            pickle.dump((X_test_features, X_train_features), fid)

        # Standardize feature vector, fit the scaler using both train and test set
        X_temp = np.vstack((X_test_features, X_train_features)).astype(np.float64)
#        del X_test_features
#        del X_train_features
        self.X_scaler = StandardScaler(copy=False).fit(X_temp)
#        with open(dump_file_path, "rb") as fid:
#            print("Reading features from", dump_file_path)
#            X_test_features, X_train_features = pickle.load(fid)

        self.X_test_scaled = self.X_scaler.transform(X_test_features)
        self.X_train_scaled = self.X_scaler.transform(X_train_features)

    def train(self, C=1.0):
        """Train the classifier"""
        self.svc = LinearSVC(C=C, max_iter=50000, verbose=True)
        start_time = time.time()
        self.svc.fit(self.X_train_scaled, self.y_train)
        end_time = time.time()
        print(round(end_time - start_time, 4), "seconds to train")

        # Validate
        test_accuracy = self.svc.score(self.X_test_scaled, self.y_test)
        print("Test Accuracy of SVC = ", round(test_accuracy, 4))

        return self.svc, self.X_scaler, self.feature_extractor_args, test_accuracy

    def demo_predict(self, n_examples=6):
        start_time = time.time()
        indices = np.random.randint(len(self.X_test_scaled), size=n_examples)
        print("My SVC predicts: ", self.svc.predict(self.X_test_scaled[indices]))
        print("For these", n_examples, "labels: ", self.y_test[indices])
        end_time = time.time()
        print(round(end_time - start_time, 4), "seconds to predict", n_examples, "examples")

    def show_false_positives(self):
        background_files = self.X_test_files[np.where(self.y_test == 0)]
        background_features = extract_features_from_files(background_files, **self.feature_extractor_args)
        X = np.array(background_features, dtype=np.float64)
        scaled_X = self.X_scaler.transform(X)
        confidences = self.svc.decision_function(scaled_X)
        indices = np.where(confidences >= 0.0)[0].astype(np.int)
        files = np.array(background_files)[indices]
        print("Num FP", len(files), "out of", len(background_files), "background files")
        print(f"Confidence score min: {np.min(confidences):.4f}, max: {np.max(confidences):.4f}, "
              f"avg: {np.mean(confidences):.4f}")
        for file, confidence in zip(files, confidences[indices]):
            print(f"The confidence score of {file} is {confidence:.04f}")
            cv2.imshow("False Positive", cv2.imread(file))
            k = cv2.waitKey() & 0xff
            if k == 27:  # esc
                break
        cv2.destroyAllWindows()

    def show_false_negatives(self):
        foreground_files = self.X_test_files[np.where(self.y_test == 1)]
        foreground_features = extract_features_from_files(foreground_files, **self.feature_extractor_args)
        X = np.array(foreground_features, dtype=np.float64)
        scaled_X = self.X_scaler.transform(X)
        confidences = self.svc.decision_function(scaled_X)
        indices = np.where(confidences < 0.0)[0].astype(np.int)
        files = np.array(foreground_files)[indices]
        print("Num FN", len(files), "out of", len(foreground_files), "vehicle files")
        print(f"Confidence score min: {np.min(confidences):.4f}, max: {np.max(confidences):.4f}, "
              f"avg: {np.mean(confidences):.4f}")
        for file, confidence in zip(files, confidences[indices]):
            print(f"The confidence score of {file} is {confidence:.04f}")
            cv2.imshow("False Negatives", cv2.imread(file))
            k = cv2.waitKey() & 0xff
            if k == 27:  # esc
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.extract_features()

    #for C in np.exp2(range(-5, 15)):
    C = 1.0
    print("Training with C =", C)
    classifier, feature_scaler, feature_extractor_args, _ = trainer.train(C=C)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dump_file_path = f"../output/classifier{timestamp}.p"
    with open(dump_file_path, "wb") as fid:
        print("Writing classifier to", dump_file_path)
        pickle.dump((classifier, feature_scaler, feature_extractor_args), fid)
#    trainer.demo_predict()
#    trainer.show_false_negatives()
#    trainer.show_false_positives()
