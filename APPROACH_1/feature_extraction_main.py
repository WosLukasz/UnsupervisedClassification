import cv2
import os
import copyUtils.copyDirectoryUtils as copy_utils
import feature_extarction.hog as hog
import joblib
from normalization.resizer_to_boundary import resize as resize
from sklearn.utils import shuffle
import numpy as np
import APPROACH_1.params as params
import feature_extarction.bov as bov
import feature_extarction.extractor as feature_extraction

main_data_directory = params.images_main_directory
main_out_data_directory = params.extracted_features_directory
dim_min = params.dim_min
dim_max = params.dim_max
batchSize = params.batchSize
dirs = os.listdir(main_data_directory)
clusters = len(dirs) # można też params.clusters

train_images_array = []
train_images_category_array = []


print("[Start] Start getting data paths...")
for dir in dirs:
    if copy_utils.is_system_file(dir):
        continue
    print("Getting data paths from class " + dir.__str__())
    source_dir_path = os.path.join(main_data_directory, dir).__str__()
    train_directory_path = os.path.join(source_dir_path, "train").__str__()
    test_directory_path = os.path.join(source_dir_path, "test").__str__()
    train_directory_file_names = copy_utils.get_file_names(train_directory_path)
    test_directory_file_names = copy_utils.get_file_names(test_directory_path)

    for file in train_directory_file_names:
        path_to_file = os.path.join(train_directory_path, file).__str__()
        train_images_array.append(path_to_file)
        train_images_category_array.append(dir.__str__())

print("[Stop] Start getting data paths...")

train_images_array, train_images_category_array = shuffle(np.array(train_images_array), np.array(train_images_category_array))
copy_utils.create_directory_if_not_exists(main_out_data_directory)

bov_md = None
bov_path = os.path.join(params.models_direcory, "bov_dict.model").__str__()

if params.train_bov_dict:
    print("[Start] Start creating bov dictionary...")
    bov_md = bov.BOV(no_clusters=params.bov_clusters)
    bov_k_means = bov_md.trainModel(train_images_array)
    joblib.dump(bov_k_means, bov_path)
    print("[Stop] Stop creating bov dictionary...")
else:
    print("[Start] Start getting bov dictionary from file...")
    bov_md = bov.BOV(no_clusters=params.bov_clusters)
    bov_k_means = joblib.load(bov_path)
    bov_md.loadModel(bov_k_means)
    print("[Stop] Stop getting bov dictionary from file...")

print("[Start] Start getting features in batches...")

features = []
answers = []
j = 0

for i in range(train_images_array.size):
    feature = feature_extraction.get_features(train_images_array[i], bov_md)
    features.append(feature)
    answers.append(train_images_category_array[i])
    if (i + 1) % batchSize == 0 or i + 1 == train_images_array.size:
        filename = j.__str__() + '.pkl'
        j = j + 1
        file_path = os.path.join(main_out_data_directory, filename).__str__()
        joblib.dump((features, answers), file_path)
        features = []
        answers = []

print("[Stop] Stop getting features in batches...")
