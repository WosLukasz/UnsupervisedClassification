import cv2
import YOLOObjectDetection.darknet as darknet
import YOLOObjectDetection.utils as yolo_utils


def print_objects(boxes):
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, cls_id, cls_conf))

cfg_file = "../yolov3.cfg"
weight_file = "../yolov3.weights"


m = darknet.Darknet(cfg_file)
m.load_weights(weight_file)


img = cv2.imread('C:/Users/wolukasz/Desktop/PG/II stopien/Praca Magisterska/src/UnsupervisedClassification/test/7.jpg')
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(original_image, (m.width, m.height))
iou_tresh = 0.4
nms_tresh = 0.6

boxes = yolo_utils.detect_objects(m, resized_image, iou_tresh, nms_tresh)
print_objects(boxes)
yolo_utils.plot_boxes(img, boxes)


