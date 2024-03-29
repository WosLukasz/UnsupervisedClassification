images_main_directory = "D:/dataset/selected8-easy-augmented" # 1  path to train data direcotory
extracted_features_directory = "D:/dataset/test6Out" # 2
selected_features_directory = "D:/dataset/test7Out" # 3
predicted_clusters_directory = "D:/dataset/test8Out" # 4
models_direcory = "D:/dataset/modelss" # path to direcotory with models
test_dataset_direcory = "D:/dataset/selected8-easy-augmented" # path to test data direcotory
batchSize = 100 # batchSize > clusters && batchSize > selected_features_number # batches in k_means and PCA
dim_min = 250 # scaling images
dim_max = 500 # scaling images
clusters = 8 # classes / clusters count
bov_clusters = 100 # using in BOV clustering
train_bov_dict = True
selected_features_number = 100 # min(n_features_in, n_samples) using in PCA