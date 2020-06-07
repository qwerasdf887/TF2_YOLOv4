import xml.etree.cElementTree as ET
import os
import cv2
import numpy as np
from .img_aug import rand_aug_image

def load_data(xml_path, classes, **kwargs):
    #load xml data
    tree = ET.parse(xml_path)
    root = tree.getroot()

    #load image
    img = cv2.imread(root.find('path').text)

    loc_list = []

    for obj in root.iter('object'):
        #難易度
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        
        #標記內容不再指定類別中 or 困難度=1則跳過該box
        if cls not in classes or int(difficult) == 1:
            continue
    
        #名稱對應的label index
        cls_id = classes.index(cls)
    
        #找到bounding box的兩個座標
        loc = obj.find('bndbox')
    
        x_min = int(loc.find('xmin').text)
        y_min = int(loc.find('ymin').text)
        x_max = int(loc.find('xmax').text)
        y_max = int(loc.find('ymax').text)
        
        loc_list.append([x_min, y_min, x_max, y_max, cls_id])
    
    
    loc_list = np.array(loc_list, dtype='float32')

    return rand_aug_image(img, loc_list, **kwargs)