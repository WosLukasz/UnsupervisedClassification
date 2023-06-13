import cv2

# int nfeatures = 0 The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
# int nOctaveLayers = 3 The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
# double contrastThreshold = 0.04 	The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
# double edgeThreshold = 10 The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
# double sigma = 1.6 	The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.


def create_sift_model(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma):
    return cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)


def create_sift_model(nfeatures):
    return cv2.xfeatures2d.SIFT_create(nfeatures)


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def get_features(sift, img):
    img_gray = to_gray(img)
    key_points, descriptors = sift.detectAndCompute(img_gray, None)
    if descriptors is not None:
        flatten_descriptors = descriptors.flatten()
        return flatten_descriptors

    return []

#
#
#
# import argparse as ap
# import cv2
# import numpy as np
# import os
# #from sklearn.externals import joblib
# from scipy.cluster.vq import *
#
# from sklearn import preprocessing
# from feature_extarction.rootsift import RootSIFT
# import math
#
# train_path = "D:/dataset/test/181/train"
#
# training_names = os.listdir(train_path)
#
# numWords = 100
#
# image_paths = []
# for training_name in training_names:
#     image_path = os.path.join(train_path, training_name)
#     image_paths += [image_path]
#
# # # Create feature extraction and keypoint detector objects
# # fea_det = cv2.FeatureDetector_create("SIFT")
# # des_ext = cv2.DescriptorExtractor_create("SIFT")
#
# sift_det = cv2.xfeatures2d.SIFT_create()
#
# # List where all the descriptors are stored
# des_list = []
#
# for i, image_path in enumerate(image_paths):
#     im = cv2.imread(image_path)
#
#     gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#     kpts, des = sift_det.detectAndCompute(gray, None)
#
#     #kpts = fea_det.detect(im)
#     #kpts, des = des_ext.compute(im, kpts)
#     # rootsift
#     # rs = RootSIFT()
#     # des = rs.compute(kpts, des)
#     des_list.append((image_path, des))
#
#
# # Stack all the descriptors vertically in a numpy array
# descriptors = des_list[0][1]
# for image_path, descriptor in des_list[1:]:
#     descriptors = np.vstack((descriptors, descriptor))
#
#
# print(descriptors.shape)
#
# # Perform k-means clustering
# voc, variance = kmeans(descriptors, numWords, 1)
#
# # Calculate the histogram of features
# im_features = np.zeros((len(image_paths), numWords), "float32")
# for i in range(len(image_paths)):
#     words, distance = vq(des_list[i][1], voc)
#     for w in words:
#         im_features[i][w] += 1
#
# # Perform Tf-Idf vectorization
# nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
# idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
#
# # Perform L2 normalization
# im_features = im_features * idf
# im_features = preprocessing.normalize(im_features, norm='l2')
#
# print(im_features.shape)
# print(im_features)
#
# print(image_paths.shape)
# print(image_paths)
#
# print(idf.shape)
# print(idf)
#
# print(voc.shape)
# print(voc)
#
# print(numWords.shape)
# print(numWords)
#
# #joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)