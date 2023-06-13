import os
import copyUtils.copyDirectoryUtils as copy_utils
from sklearn.decomposition import IncrementalPCA
import joblib
import numpy as np
import APPROACH_1.params as params

main_data_directory = params.extracted_features_directory
main_data_out_directory = params.models_direcory
main_dumps_out_directory = params.selected_features_directory
pca_model_path = os.path.join(main_data_out_directory, "pca.model").__str__()
batchSize = params.batchSize
n_features = params.selected_features_number
dumps = os.listdir(main_data_directory)
copy_utils.create_directory_if_not_exists(main_data_out_directory)
copy_utils.create_directory_if_not_exists(main_dumps_out_directory)

pca = IncrementalPCA(n_components=n_features)

print("[Start] Start selection of features...")
for dump in dumps:
    if copy_utils.is_system_file(dump):
        continue

    source_path = os.path.join(main_data_directory, dump).__str__()
    features, _ = joblib.load(source_path)
    if np.shape(features)[0] < n_features: # jest na to bug zgloszony: https://github.com/scikit-learn/scikit-learn/issues/12234 pomyslec moze nad czyms innym
        print("Batch is to small " + np.shape(features)[0].__str__())
        continue
    pca.partial_fit(features)


print("[Stop] Stop selection of features...")

joblib.dump(pca, pca_model_path)

print("[Start] Start selection of features for dumps...")
for dump in dumps:
    if copy_utils.is_system_file(dump):
        continue

    print("Selecting features for dump" + dump.__str__())
    source_path = os.path.join(main_data_directory, dump).__str__()
    features, answers = joblib.load(source_path)
    transformed_features = pca.transform(features)
    out_path = os.path.join(main_dumps_out_directory, dump.__str__()).__str__()
    joblib.dump((transformed_features, answers), out_path)


print("[Stop] Stop selection of features for dumps...")
