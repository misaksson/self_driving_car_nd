import unittest
import numpy as np
import pickle
import os
import shutil

from TrafficSignData import TrafficSignData


class TestTrafficSignData(unittest.TestCase):
    TEST_DATA_PATH = './src/test/temp/'

    @classmethod
    def setUpClass(cls):
        """Generate test data"""
        cls.n_train = 100
        cls.n_valid = 3
        cls.n_test = 3
        cls.image_shape = (32, 32, 3)
        cls.n_classes = 5
        cls.X_train = cls._generate_images(cls.n_train, cls.image_shape)
        cls.y_train = np.floor(np.random.rand(cls.n_train) * cls.n_classes)
        cls.X_valid = cls._generate_images(cls.n_valid, cls.image_shape)
        cls.y_valid = np.zeros(cls.n_valid)
        cls.X_test = cls._generate_images(cls.n_test, cls.image_shape)
        cls.y_test = np.zeros(cls.n_test)

        # This directory is assumed to not already exist.
        assert(os.path.exists(cls.TEST_DATA_PATH) is False)
        os.makedirs(cls.TEST_DATA_PATH)

        with open(os.path.join(cls.TEST_DATA_PATH, 'train.p'), 'wb') as f:
            pickle.dump({'features': cls.X_train, 'labels': cls.y_train}, f)

        with open(os.path.join(cls.TEST_DATA_PATH, 'valid.p'), 'wb') as f:
            pickle.dump({'features': cls.X_valid, 'labels': cls.y_valid}, f)

        with open(os.path.join(cls.TEST_DATA_PATH, 'test.p'), 'wb') as f:
            pickle.dump({'features': cls.X_test, 'labels': cls.y_test}, f)

    def _generate_images(n_images, shape):
        images = []
        for i in range(n_images):
            image = np.round(np.random.rand(shape[0], shape[1], shape[2]) * 255)
            images.append(image)

        return images

    def test_properties(self):
        ts_data = TrafficSignData(path=TestTrafficSignData.TEST_DATA_PATH)
        self.assertEqual(ts_data.n_train, TestTrafficSignData.n_train)
        self.assertEqual(ts_data.n_valid, TestTrafficSignData.n_valid)
        self.assertEqual(ts_data.n_test, TestTrafficSignData.n_test)
        self.assertEqual(ts_data.image_shape, TestTrafficSignData.image_shape)
        self.assertEqual(ts_data.n_classes, TestTrafficSignData.n_classes)

    def test_normalized_images(self):
        ts_data = TrafficSignData(path=TestTrafficSignData.TEST_DATA_PATH)
        self._compare_raw_vs_normalized_histograms(TestTrafficSignData.X_train, ts_data.X_train)
        self._compare_raw_vs_normalized_histograms(TestTrafficSignData.X_valid, ts_data.X_valid)
        self._compare_raw_vs_normalized_histograms(TestTrafficSignData.X_test, ts_data.X_test)

    def _compare_raw_vs_normalized_histograms(self, raw_images, norm_images):
        """Compare images using histograms."""
        raw_range = (-0.5, 255.5)
        norm_range = ((raw_range[0] - 128.) / 128., (raw_range[1] - 128.) / 128.)
        for raw_image, norm_image in zip(raw_images, norm_images):
            raw_hist, _ = np.histogram(raw_image, bins=256, range=raw_range)
            norm_hist, _ = np.histogram(norm_image, bins=256, range=norm_range)
            self.assertTrue(np.array_equal(raw_hist, norm_hist))

    def test_shuffle_training_data(self):
        default = TrafficSignData(path=TestTrafficSignData.TEST_DATA_PATH)
        shuffled = TrafficSignData(path=TestTrafficSignData.TEST_DATA_PATH)
        shuffled.shuffle_training_data()

        # All but the training data should be the same.
        self.assertFalse(np.array_equal(default.X_train, shuffled.X_train))
        self.assertFalse(np.array_equal(default.y_train, shuffled.y_train))
        self.assertTrue(np.array_equal(default.X_valid, shuffled.X_valid))
        self.assertTrue(np.array_equal(default.y_valid, shuffled.y_valid))
        self.assertTrue(np.array_equal(default.X_test, shuffled.X_test))
        self.assertTrue(np.array_equal(default.y_test, shuffled.y_test))

    @classmethod
    def tearDownClass(cls):
        """Delete generated test data"""
        shutil.rmtree(cls.TEST_DATA_PATH)


if __name__ == '__main__':
    unittest.main()
