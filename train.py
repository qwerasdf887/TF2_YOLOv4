import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow_addons as tfa
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
from model.models import creat_CSPYOLO, load_weights
from model.loss import yolo_loss
from model.load_xml_data import load_data
from model.utils import preprocess_true_boxes
import os
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#設定為float16
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

train_default = {'anchors': [[12, 16], [19, 36], [40, 28],
                            [36, 75], [76, 55], [72, 146],
                            [142, 110], [192, 243], [459, 401]],
                 'anchors_mask': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 'dropblock': [False, True, True, True, True], #分別對應每個residual block後是否加入dropblock
                 'image_shape': (608, 608),
                 'num_classes': 5,
                 'score_threshold': 0.6,
                 'iou_threshold': 0.3,
                 'batch_size': 2,
                 'use_pretrain': True,
                 'drop_rate' : 0.2,
                 'block_size' : 3
                }

#default aug args:
default_aug = {'noise_prob': 0.1,
               'gasuss_mean': 0,
               'gasuss_var': 0.001,
               'rand_hug': 30,
               'rand_saturation':30,
               'rand_light': 30,
               'rot_angle': 15,
               'bordervalue': (127, 127, 127),
               'zoom_out_value': 0.7,
               'blur_kernel': (5, 5),
               'output_shape': (608, 608),
               'take_value' : 5
              }

#讀取類別，回傳類別List
def get_classes(classes_path):
    with open(classes_path) as f:
        class_name = f.readlines()
    class_name = [c.strip() for c in class_name]
    return class_name

class img_gen(Sequence):
    def __init__(self, xml_name, ann_path, classes_name, aug_dict, model_dict):
        self.xml_name = xml_name
        self.ann_path = ann_path
        self.batch = model_dict['batch_size']
        self.cls_name = classes_name
        self.aug_dict = aug_dict
        self.model_dict = model_dict

    def __len__(self):
        return int(np.ceil(len(self.xml_name) / self.batch))
    def __getitem__(self, idx):
        if idx == 0:
            np.random.shuffle(self.xml_name)
        image_data = []
        y_true = []
        batch_name = self.xml_name[idx * self.batch: (idx + 1) * self.batch]
        for name in batch_name:
            path = self.ann_path + name
            img, loc = load_data(path, self.cls_name, **self.aug_dict)
            boxes = preprocess_true_boxes(loc, **self.model_dict)
            image_data.append(img)
            y_true.append(boxes)

        
        image_data = (np.array(image_data) / 255).astype(np.float32)
        y_true = np.array(y_true)

        return image_data, y_true


if __name__ == "__main__":
    model = creat_CSPYOLO(**train_default)
    if train_default['use_pretrain']:
        custom = False if train_default['num_classes'] == 80 else True
        load_weights(model, './yolov4.weights', custom_cls=custom)
    
    model.summary()
    
    annotation_path = 'path'
    log_dir = 'path'
    classes_path = 'path'
    class_name = get_classes(classes_path)

    xmls = os.listdir(annotation_path)
    total_train = len(xmls)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath= log_dir + 'best_loss.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    verbose=1)]

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt = tfa.optimizers.MovingAverage(opt)
    model.compile(optimizer=opt, loss=yolo_loss)
    model.fit(img_gen(xmls, annotation_path, class_name, default_aug, train_default),
              steps_per_epoch= np.ceil(total_train / train_default['batch_size']),
              callbacks=callbacks, epochs=200)
    model.save_weights(log_dir + 'final_yolov4.h5')
