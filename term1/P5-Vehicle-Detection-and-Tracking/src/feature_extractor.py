import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    """Computes binned color features

    Resample image according to size argument and return the result as a 1D
    array.
    """
    features = cv2.resize(img, size).ravel()
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Computes the image histogram

    The histogram of each color channel is concatenated  into one feature array.
    """

    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def extract_features(bgr_image, color_space='BGR', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channels=[0],
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    # apply color conversion if other than 'BGR'
    if color_space == 'BGR':
        image = bgr_image
    elif color_space == 'RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LUV':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LUV)
    elif color_space == 'HLS':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS)
    elif color_space == 'YUV':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
    elif color_space == 'YCrCb':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    else:
        assert(False)  # Requested color space not implemented.

    image_features = []

    if spatial_feat:
        spatial_features = bin_spatial(image, size=spatial_size)
        image_features.append(spatial_features)

    if hist_feat:
        hist_features = color_hist(image, nbins=hist_bins)
        image_features.append(hist_features)

    if hog_feat:
        hog_features = []
        for channel in hog_channels:
            hog_features.append(get_hog_features(image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)

        image_features.append(hog_features)

    return np.concatenate(image_features)


def extract_features_from_files(image_files, **kwargs):
    """Extract features from list of image files

    [description]

    Arguments:
        image_files -- List of image files.
        **kwargs -- see extract_features() for optional arguments

    Returns:
        List of feature vectors, one fore each image.
    """
    # Iterate through the list of images
    pbar_files = tqdm(image_files)
    all_features = []
    for file in pbar_files:
        image_features = []
        bgr_image = cv2.imread(file)
        image_features = extract_features(bgr_image, **kwargs)
        all_features.append(image_features)
    return all_features
