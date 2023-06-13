import cv2
import feature_extarction.hog as hog
from normalization.resizer_to_boundary import resize as resize
import numpy as np


def get_features(path_to_file, bov_model):
    image = cv2.imread(path_to_file)
    sift_features = bov_model.getHistogramOfImage(image)
    image = resize(image)  # do wywalenia bo dane po augmentacji będą już w jednym rozmiarze
    hog_features = hog.hog_feature_vector(image)
    return np.concatenate([hog_features, sift_features])