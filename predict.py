import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import cv2
from model.models import creat_CSPYOLO, load_weights, output_result

#讀取類別，回傳類別List
def get_classes(classes_path):
    with open(classes_path) as f:
        class_name = f.readlines()
    class_name = [c.strip() for c in class_name]
    return class_name

#等比例縮放影像
def resiresize_img(image):
    h, w, _ = image.shape
    max_edge = max(608, 608)
    scale = min( max_edge / h, max_edge / w)
    h = int(h * scale)
    w = int(w * scale)
    return cv2.resize(image, (w, h))

#將樸片補至指定大小
def padding_img(image):
    h, w, _ = image.shape

    dx = int((608 - w) / 2)
    dy = int((608 - h) / 2)

    out_img = np.ones((608, 608, 3), np.uint8) * 127
    out_img[dy: dy + h, dx: dx + w, :] = image

    return out_img

def draw_img(img, boxes, cls, socre, class_name, **kwargs):
    h, w = kwargs['image_shape']
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 255), 2)
        label = class_name[cls[i]]
        cv2.putText(img, label, (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

if __name__ == "__main__":
    default = {'anchors': [[12, 16], [19, 36], [40, 28],
                           [36, 75], [76, 55], [72, 146],
                           [142, 110], [192, 243], [459, 401]],
               'anchors_mask': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
               'dropblock': [False, False, False, False, False], #分別對應每個residual block後是否加入dropblock
               'image_shape': (608, 608),
               'num_classes': 80,
               'score_threshold': 0.6,
               'iou_threshold': 0.3,
               'batch_size': 2,
               'drop_rate' : 0.2,
               'block_size' : 3
              }

    class_name = get_classes('./coco_classes.txt')

    model = creat_CSPYOLO(**default)
    load_weights(model, './yolov4.weights')
    img = cv2.imread('./kite.jpg')
    img = resiresize_img(img)
    img = padding_img(img)
    pred_img = np.expand_dims(img, axis=0).astype(np.float32) / 255

    result = model.predict(pred_img)
    boxes, class_, score = output_result(result, **default)

    draw_img(img, boxes, class_, score, class_name, **default)

    cv2.imwrite('./predict.jpg', img)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()