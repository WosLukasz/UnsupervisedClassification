import os
import copyUtils.copyDirectoryUtils as copy_utils

main_data_source_directory = "D:/dataset/processed"
dirs = os.listdir(main_data_source_directory)

print("[Start] Generating images from videos...")
for dir in dirs:
    source_dir_path = os.path.join(main_data_source_directory, dir).__str__()
    print(source_dir_path)
    sub_dirs = os.listdir(source_dir_path)
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(source_dir_path, sub_dir).__str__()
        files = os.listdir(sub_dir_path)
        for file in files:
            image_path = os.path.join(sub_dir_path, file).__str__()
            if copy_utils.is_system_file(image_path):
                continue
            new_name = dir + "." + sub_dir + "." + file
            new_name = new_name.replace(" ", "_")
            new_path = os.path.join(sub_dir_path, new_name).__str__()
            os.rename(image_path, new_path)

