import os
import cv2
from normalization.resizer_to_boundary import resize_old as resize_old
import shutil


def get_file_names(path):
    file_names = os.listdir(path)
    return list(filter(lambda x: x.endswith(('.png', '.jpg')), file_names))


def is_system_file(filename):
    return filename.endswith('.db')


def copy_directory(source_directory, destination_directory, dim_min, dim_max, force=False):
    if not os.path.isdir(destination_directory) or force:
        if os.path.isdir(destination_directory) and force:
            shutil.rmtree(destination_directory)
        print("[Start] Copy dataset...")
        original_dirs = os.listdir(source_directory)
        os.mkdir(destination_directory)
        for dir in original_dirs:
            new_dir_path =  os.path.join(destination_directory, dir).__str__()
            os.mkdir(new_dir_path)
            original_dir_path = os.path.join(source_directory, dir).__str__()
            new_file_names = get_file_names(original_dir_path)
            for new_file_name in new_file_names:
                img = cv2.imread(os.path.join(original_dir_path, new_file_name).__str__())
                new_image = resize_old(img, dim_min, dim_max)
                cv2.imwrite(os.path.join(new_dir_path, new_file_name).__str__(), new_image)
        print("[Stop] Copy dataset...")
    else:
        print("[INFO] Dataset already exist. Skipped.")


def create_directory_if_not_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Error: Creating directory of data')


def copy_file(src_path, dest_path):
    shutil.copy(src_path, dest_path)


def save_predicted_clusters(main_out_data_directory, test_predicted, image_array):
    create_directory_if_not_exists(main_out_data_directory)
    for i in range(0, len(test_predicted)):
        file_path = image_array[i]
        label = test_predicted[i]
        out_class_directory = os.path.join(main_out_data_directory, str(label)).__str__()
        create_directory_if_not_exists(out_class_directory)
        copy_file(file_path, out_class_directory)