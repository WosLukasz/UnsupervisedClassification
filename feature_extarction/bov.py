import cv2
import numpy as np
from glob import glob
import argparse
from feature_extarction.bov_helpers import *
#from matplotlib import pyplot as plt
import APPROACH_1.params as params
from scipy.cluster.vq import *
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans

class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.k_means = None

    def trainModel(self, train_images_paths):
        dico = []
        for im_path in train_images_paths:
            im = cv2.imread(im_path)
            kp, des = self.im_helper.features(im)
            # print(des)
            if des is None:
                continue
            for d in des:
                dico.append(d)
            #
            # print(len(des))
            #self.descriptor_list.append(des)
            #self.descriptor_list.append((im_path, des))

        batch_size = np.size(params.batchSize)
        kmeans = MiniBatchKMeans(n_clusters=self.no_clusters, batch_size=batch_size, verbose=0).fit(dico)
        self.k_means = kmeans
        return self.k_means

    def loadModel(self, k_means_model):
        self.k_means = k_means_model

    def getHistogramOfImages(self, images_paths):
        histo_list = []
        for img_path in images_paths:
            img = cv2.imread(img_path)
            kp, des = self.im_helper.features(img)

            histo = np.zeros(self.no_clusters)
            nkp = np.size(kp)
            if des is not None:
                for d in des:
                    idx = self.k_means.predict([d])
                    histo[idx] += 1 / nkp   # Because we need normalized histograms, I prefere to add 1/nkp directly

            histo_list.append(histo)


        return histo_list


    def getHistogramOfImage(self, img):
        kp, des = self.im_helper.features(img)

        histo = np.zeros(self.no_clusters)
        nkp = np.size(kp)
        if des is not None:
            for d in des:
                idx = self.k_means.predict([d])
                histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

        return histo

        # descriptors = self.descriptor_list[0][1]
        #
        # for image_path, descriptor in self.descriptor_list[1:]:
        #     descriptors = np.vstack((descriptors, descriptor))
        #
        #
        # voc, variance = kmeans(descriptors, self.no_clusters, 1)
        #
        # # Calculate the histogram of features
        # im_features = np.zeros((len(self.train_images_paths), self.no_clusters), "float32")
        # for i in range(len(self.train_images_paths)):
        #     words, distance = vq(self.descriptor_list[i][1], voc)
        #     for w in words:
        #         im_features[i][w] += 1
        #
        # # Perform Tf-Idf vectorization
        # nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
        # idf = np.array(np.log((1.0 * len(self.train_images_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
        #
        # # Perform L2 normalization
        # im_features = im_features * idf
        # im_features = preprocessing.normalize(im_features, norm='l2')
        #
        # return im_features



        # print(np.shape(self.descriptor_list))
        # self.trainImageCount = len(self.train_images_paths)
        # bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        # self.bov_helper.cluster()
        # self.bov_helper.developVocabulary(n_images=self.trainImageCount, descriptor_list=self.descriptor_list)
        # return self.bov_helper.standardize()


    # def recognize(self, test_img, test_image_path=None):
    #
    #     """
    #     This method recognizes a single image
    #     It can be utilized individually as well.
    #     """
    #
    #     kp, des = self.im_helper.features(test_img)
    #
    #     # generate vocab for test image
    #     vocab = np.array([[0 for i in range(self.no_clusters)]])
    #     # locate nearest clusters for each of
    #     # the visual word (feature) present in the image
    #
    #     # test_ret =<> return of kmeans nearest clusters for N features
    #     test_ret = self.bov_helper.kmeans_obj.predict(des)
    #
    #     for each in test_ret:
    #         vocab[0][each] += 1
    #
    #
    #     # Scale the features
    #     vocab = self.bov_helper.scale.transform(vocab)
    #
    #     # predict the class of the image
    #     lb = self.bov_helper.clf.predict(vocab)
    #
    #     return lb

    # def testModel(self):
    #     self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)
    #
    #     predictions = []
    #
    #     for word, imlist in self.testImages.iteritems():
    #         for im in imlist:
    #
    #             cl = self.recognize(im)
    #
    #             predictions.append({
    #                 'image': im,
    #                 'class': cl,
    #                 'object_name': self.name_dict[str(int(cl[0]))]
    #             })
    #
    #     for each in predictions:
    #         # cv2.imshow(each['object_name'], each['image'])
    #         # cv2.waitKey()
    #         # cv2.destroyWindow(each['object_name'])
    #         #
    #         plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
    #         plt.title(each['object_name'])
    #         plt.show()



