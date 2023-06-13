import json
import cv2
import normalization.image_utils as image_utils
import copyUtils.copyDirectoryUtils as copyUtils


def create_path(main_out, vals):
    return main_out + '/' + vals[0] + '/' + vals[1] + '/'

main_out = 'D:/dataset/master'
metadata_file = open("D:/dataset/master_out/vott-json-export/master-export.json", "r")
metadata = json.loads(metadata_file.read())

assets = metadata["assets"]
j = 0
for (k, v) in assets.items():
    path = v["asset"]["path"]
    path = path[5:]
    file_name = v["asset"]["name"]
    print("Processing file: " + file_name)
    vals = file_name.split(".")
    new_path = create_path(main_out, vals)
    copyUtils.create_directory_if_not_exists(new_path)
    regions = v["regions"]
    iterator = 0
    for region in regions:
        x = int(region["boundingBox"]["left"])
        y = int(region["boundingBox"]["top"])
        w = int(region["boundingBox"]["width"])
        h = int(region["boundingBox"]["height"])
        img = cv2.imread(path)
        cropped_img = image_utils.crop_image_vott(img, x, y, w, h)
        cropped_path = new_path + str(iterator) + file_name
        print("Saving file: " + cropped_path)
        cv2.imwrite(cropped_path, cropped_img)
        iterator = iterator + 1



