import pickle
import numpy as np
import matplotlib.pyplot as plt


class TrafficSignData(object):
    def __init__(self, path="./data/", demo_mode=False):
        self.demo_mode = demo_mode
        self._load_data(path)
        if self.demo_mode:
            self._show_examples()
        self._extract_properties()
        self._normalize_input_data()
        self._balance_training_data()
        self._extract_properties()  # Repeat due to updated data.

    def _load_data(self, path):
        from os.path import join
        self.X_train, self.y_train = self._load_pickled_file(join(path, "train.p"))
        self.X_valid, self.y_valid = self._load_pickled_file(join(path, "valid.p"))
        self.X_test, self.y_test = self._load_pickled_file(join(path, "test.p"))

    def _load_pickled_file(self, file_path):
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            return data['features'], data['labels']

    def _show_examples(self):
        """
        Show examples of each image class/label
        """
        labels, label_indices, label_counts = np.unique(self.y_train, return_index=True, return_counts=True)
        plt.figure(figsize=(15, 20))
        for idx in range(len(labels)):
            ax = plt.subplot(9, 5, idx + 1)
            ax.imshow(self.X_train[label_indices[idx]])
            ax.axis('off')
            ax.set_title(f"label {labels[idx]}: {label_counts[idx]} images")

        plt.show()

    def _extract_properties(self):
        # Number of training examples
        self.n_train = len(self.y_train)

        # Number of validation examples
        self.n_valid = len(self.y_valid)

        # Number of testing examples.
        self.n_test = len(self.y_test)

        # Shape of an traffic sign image
        self.image_shape = self.X_train[0].shape

        # Find list of all labels and count of each label in the training set
        labels, _, label_counts = np.unique(self.y_train, return_index=True, return_counts=True)

        # How many unique classes/labels there are in the dataset.
        self.n_classes = len(labels)

        # Weights punishing classes with many training examples.
        self.class_weights = np.log(self.n_train / np.max((label_counts, np.ones(label_counts.shape)), axis=0)) + 1.

        if self.demo_mode:
            print("Number of training examples =", self.n_train)
            print("Number of validation examples =", self.n_valid)
            print("Number of testing examples =", self.n_test)
            print("Image data shape =", self.image_shape)
            print("Number of classes =", self.n_classes)

    def _normalize_input_data(self):
        raw_image = np.copy(self.X_train[0])
        self.X_train = self._normalize_images(self.X_train)
        self.X_valid = self._normalize_images(self.X_valid)
        self.X_test = self._normalize_images(self.X_test)

        if self.demo_mode:
            # Plot histogram before and after normalization for one image and color channel
            # to check that the result seems reasonable.
            plt.figure(figsize=(15, 10))

            # Raw image
            ax = plt.subplot(2, 2, 1)
            # Make sure pixel values are in the middle of each bin to avoid numerical errors.
            raw_hist_range = (-0.5, 255.5)
            raw_hist, _, _ = ax.hist(raw_image[:, :, 0].flatten(), bins=256, range=raw_hist_range)
            ax.axis([raw_hist_range[0], raw_hist_range[1], 0, np.max(raw_hist) + 5])
            ax.set_title("Raw image histogram")

            # Normalized image
            ax = plt.subplot(2, 2, 2)
            norm_hist_range = ((raw_hist_range[0] - 128.) / 128., (raw_hist_range[1] - 128.) / 128.)
            norm_hist, _, _ = ax.hist(self.X_train[0][:, :, 0].flatten(), bins=256, range=norm_hist_range)
            ax.axis([norm_hist_range[0], norm_hist_range[1], 0, np.max(norm_hist) + 5])
            ax.set_title("Normalized image histogram")
            assert(np.array_equal(raw_hist, norm_hist))

            ax = plt.subplot(2, 2, 3)
            ax.imshow(raw_image)
            ax.set_title("Raw image")
            ax = plt.subplot(2, 2, 4)
            ax.imshow(self.X_train[0])
            ax.set_title("Normalized image")
            plt.show()

    def _normalize_images(self, images):
        """
        Normalize 8 bit input images to range -1.0 to 1.0.
        """
        result = []
        for image in images:
            assert(image.max() <= 255)
            assert(image.min() >= 0)
            result.append(np.divide(np.subtract(image, 128.0), 128.0))

        return np.array(result)

    def _balance_training_data(self):
        """Balance training data by oversampling.

        Replicate images from classes with few examples such that all gets the same count.
        """
        sorting_indices = np.argsort(self.y_train)
        self.y_train = self.y_train[sorting_indices]
        self.X_train = self.X_train[sorting_indices]

        labels, label_indices, label_counts = np.unique(self.y_train, return_index=True, return_counts=True)
        max_count = np.max(label_counts).astype(np.float64)
        balanced_y_train = np.array([])
        balanced_X_train = np.empty((1, 32, 32, 3))
        for label, idx, count in zip(labels, label_indices, label_counts):
            # labels
            tiled = np.tile(self.y_train[idx: idx + count], np.ceil(max_count / count).astype(np.int))
            tiled = tiled[:max_count.astype(np.int)]
            balanced_y_train = np.concatenate((balanced_y_train, tiled))

            # images
            tiled = np.tile(self.X_train[idx: idx + count, :, :, :],
                            (np.ceil(max_count / count).astype(np.int), 1, 1, 1))
            tiled = tiled[:max_count.astype(np.int), :, :, :]
            balanced_X_train = np.concatenate((balanced_X_train, tiled), axis=0)

        self.y_train = balanced_y_train
        self.X_train = balanced_X_train[1:, :, :, :]
        if self.demo_mode:
            # Show that there now is an equal number of images per class.
            self._show_examples()

    def shuffle_training_data(self):
        from sklearn.utils import shuffle
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
