import sys
import cv2
import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_extractor import extract_features_from_files


class Trainer(object):
    """Vehicle classifier trainer

    Loads image examples, extract features and train a classifier.
    """

    default_feature_extractor_args = {'color_space': 'HSV',
                                      'spatial_size': (32, 32),
                                      'hist_bins': 32,
                                      'orient': 9,
                                      'pix_per_cell': 8,
                                      'cell_per_block': 2,
                                      'hog_channel': 'ALL',
                                      'spatial_feat': True,
                                      'hist_feat': True,
                                      'hog_feat': True,
                                      }


    def __init__(self):
        self._find_training_images()

    def _find_training_images(self):
        """Locates a list of training example files

        Recursive search of image examples in the training_data folder.
        """
        image_files = glob.glob("../training_data/**/*.png", recursive=True)
        self.vehicle_files = []
        self.background_files = []
        for image_file in image_files:
            if "non-vehicles" in image_file:
                self.background_files.append(image_file)
            elif "vehicles" in image_file:
                self.vehicle_files.append(image_file)
            else:
                assert(False)

    def train(self, feature_extractor_args=None):
        """Train the classifier"""

        if feature_extractor_args is not None:
            self.feature_extractor_args = feature_extractor_args
        else:
            self.feature_extractor_args = Trainer.default_feature_extractor_args

        vehicle_features = extract_features_from_files(self.vehicle_files, **self.feature_extractor_args)
        background_features = extract_features_from_files(self.background_files, **self.feature_extractor_args)

        print("Extracted features from ", len(self.vehicle_files),
              "vehicle files, utilizing", sys.getsizeof(vehicle_features), "bytes")
        print("Extracted features from ", len(self.background_files),
              "background files, utilizing", sys.getsizeof(background_features), "bytes")

        # Standardize feature vector
        X = np.vstack((vehicle_features, background_features)).astype(np.float64)
        self.X_scaler = StandardScaler().fit(X)
        scaled_X = self.X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(background_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        # Train
        self.svc = LinearSVC()
        start_time = time.time()
        self.svc.fit(X_train, y_train)
        end_time = time.time()
        print(round(end_time - start_time, 4), "seconds to train")

        # Validate
        print("Test Accuracy of SVC = ", round(self.svc.score(self.X_test, self.y_test), 4))

        return self.svc, self.X_scaler, self.feature_extractor_args

    def demo_predict(self, n_examples=6):
        start_time = time.time()
        indices = np.random.randint(len(self.X_test), size=n_examples)
        print("My SVC predicts: ", self.svc.predict(self.X_test[indices]))
        print("For these", n_examples, "labels: ", self.y_test[indices])
        end_time = time.time()
        print(round(end_time - start_time, 4), "seconds to predict", n_examples, "examples")

    def show_false_positives(self):
        background_features = extract_features_from_files(self.background_files, **self.feature_extractor_args)
        X = np.array(background_features, dtype=np.float64)
        scaled_X = self.X_scaler.transform(X)
        confidences = self.svc.decision_function(scaled_X)
        indices = np.where(confidences >= 0.0)[0].astype(np.int)
        files = np.array(self.background_files)[indices]
        print("Num FP", len(files), "out of", len(self.background_files), "background files")
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
        vehicle_features = extract_features_from_files(self.vehicle_files, **self.feature_extractor_args)
        X = np.array(vehicle_features, dtype=np.float64)
        scaled_X = self.X_scaler.transform(X)
        confidences = self.svc.decision_function(scaled_X)
        indices = np.where(confidences < 0.0)[0].astype(np.int)
        files = np.array(self.vehicle_files)[indices]
        print("Num FN", len(files), "out of", len(self.vehicle_files), "vehicle files")
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
    trainer.train()
    trainer.demo_predict()
    trainer.show_false_negatives()
    trainer.show_false_positives()
