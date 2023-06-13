import os
import cv2
import copyUtils.copyDirectoryUtils as copy_utils
import normalization.image_utils as image_utils
import re


main_data_source_directory = "./videos"
main_data_destination_directory = "D:\dataset\processed"
dirs = os.listdir(main_data_source_directory)

print("Creating main directory if not exists...")
copy_utils.create_directory_if_not_exists(main_data_destination_directory)

print("[Start] Generating images from videos...")
for dir in dirs:
    source_dir_path = os.path.join(main_data_source_directory, dir).__str__()
    dest_dir_path = os.path.join(main_data_destination_directory, dir).__str__()

    if copy_utils.is_system_file(source_dir_path):
        continue

    print("Creating class output directory if not exists [ " + dest_dir_path + " ]")
    copy_utils.create_directory_if_not_exists(dest_dir_path)

    videos_names = os.listdir(source_dir_path)

    for video_name in videos_names:
        video_path = os.path.join(source_dir_path, video_name).__str__()
        direcotry_name = re.sub('\.mp4$', '', video_name)
        images_dest_directory_path = os.path.join(dest_dir_path, direcotry_name).__str__()
        if os.path.exists(images_dest_directory_path) or copy_utils.is_system_file(video_name):
            continue

        copy_utils.create_directory_if_not_exists(images_dest_directory_path)

        print("Proccesing video [ " + video_name + "]")
        cap = cv2.VideoCapture(video_path)

        currentFrame = 0
        while (True):
            ret, frame = cap.read()
            if frame is None:
                break
            out_image_path = os.path.join(images_dest_directory_path, str(currentFrame) + '.jpg').__str__()
            cv2.imwrite(out_image_path, frame)
            currentFrame += 1

print("[Stop] Generating images from videos.")