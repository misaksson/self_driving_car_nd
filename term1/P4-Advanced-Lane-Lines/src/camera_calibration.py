import cv2
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random


class Calibrate(object):
    def __init__(self, camera_cal_path="../camera_cal/"):
        self.path = camera_cal_path
        self._load_images()
        self._calibrate()

    def _load_images(self):
        self.file_names = os.listdir(self.path)
        self.file_names.sort(key=lambda x: float(x.strip('calibration').strip('jpg')))  # sort files by numbers

        images = []
        for name in self.file_names:
            bgr_image = cv2.imread(os.path.join(self.path, name))
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            images.append(rgb_image)

        self.images = images

    def _calibrate(self):
        height, width, _ = self.images[0].shape

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        self.corner_images = []
        for idx, image in enumerate(self.images):
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Try to find chessboard, starting with the expected number of corners and then decrease
            # the number of corners in each iteration until the chessboard is found.
            for n_col, n_row in itertools.product(*[range(9, 6, -1), range(6, 2, -1)]):
                ret, corners = cv2.findChessboardCorners(gray_image, (n_col, n_row), None)
                if ret:
                    # Prepare object points for found corners:
                    # (0, 0, 0), (1, 0, 0), (2, 0, 0) ....,(n_col - 1, n_row - 1, 0)
                    objp = np.zeros((n_row * n_col, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:n_col, 0:n_row].T.reshape(-1, 2)
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    corner_image = np.copy(image)
                    cv2.drawChessboardCorners(corner_image, (n_col, n_row), corners, ret)
                    self.corner_images.append(corner_image)
                    break
            else:
                print(f"Unable to use {self.file_names[idx]} for calibration.")
                corner_image = np.copy(image)
                cv2.line(corner_image, (0, 0), (width - 1, height - 1), color=(255, 0, 0), thickness=15)
                cv2.line(corner_image, (width - 1, 0), (0, height - 1), color=(255, 0, 0), thickness=15)
                self.corner_images.append(corner_image)

        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def demo_chessboard(self, figsize=(8, 8), output_path=None):
        n_cols = 4
        n_rows = math.ceil(len(self.corner_images) / n_cols)
        fig = plt.figure(figsize=figsize)
        for idx, (corner_image, image, name) in enumerate(zip(self.corner_images, self.images, self.file_names)):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            ax.imshow(corner_image)
            ax.axis('off')
            ax.set_title(name, fontsize=10)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.01)
        fig.subplots_adjust(left=0., right=1., top=0.98, bottom=0.)

        if output_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output_path, 'chessboard.png'))

    def demo_undistort(self, images=None, figsize=(7, 15), output_path=None):
        if images is not None:
            n_demo_images = len(images)
            demo_images = images
        else:
            # Show undistorted calibration images.
            n_demo_images = min(len(self.images), 5)
            demo_images = random.sample(self.images, n_demo_images)

        fig, ax = plt.subplots(n_demo_images, 2, figsize=figsize)
        fig.tight_layout()
        for idx, image in enumerate(demo_images):
            ax[idx, 0].imshow(image)
            ax[idx, 0].axis('off')
            ax[idx, 1].imshow(self.undistort(image))
            ax[idx, 1].axis('off')

        plt.subplots_adjust(left=0., right=1, top=1., bottom=0.)
        if output_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output_path, 'undistort.png'))


if __name__ == '__main__':
    cal = Calibrate()
    cal.demo_chessboard()
    cal.demo_undistort()
