import cv2
import numpy as np
from grid_generator import GridGenerators
from classifier import Classifier
from draw import Draw

from colormap import cmap_builder


confidence_cmap = cmap_builder('yellow', 'lime (w3c)', 'cyan')
confidence_range = np.array([0.0, 5.0])
draw = Draw(cmap=confidence_cmap, value_range=confidence_range)

bgr_image = cv2.imread("../images/bbox-example-image.jpg")
draw_image = np.copy(bgr_image)
draw.colorbar(draw_image, ticks=np.arange(confidence_range[0], confidence_range[1] + 1))
search_windows = GridGenerators(bgr_image)
classifier = Classifier(force_train=False)

for search_window in search_windows.next():
    patch = cv2.resize(bgr_image[search_window.top:search_window.bottom,
                                 search_window.left:search_window.right], (64, 64))
    prediction, confidence = classifier.classify(patch)
    if prediction:
        draw.box(draw_image, box=search_window, value=confidence)
cv2.imshow("Input image", bgr_image)
cv2.imshow("Detection image", draw_image)
cv2.waitKey()
cv2.destroyAllWindows()
