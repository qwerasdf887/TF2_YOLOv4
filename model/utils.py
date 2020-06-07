import tensorflow as tf
import numpy as np
import cv2
import math

'''
DIoU:
    IoU - (EU_dis(gt_c, pred_c)^2) / c^2
    gt_c : ground truth box center
    pred_c : predict box center
    c : diagonal length of the smallest enclosing box covering the two boxes.

CIoU:
    IoU - DIou - alpha*v
'''
'''
一張圖有N個bounding box，使用多個anchors配對的情況下，會有K個配對成功。
計算K個pred bounding box 與 ground truth bounding box的IoU，
可看成每個預測的bounding與全部ground truth bounding box算IoU後取最大值。
'''
def box_iou(box_1, box_2):
    '''
    Args:
         box_1(i.e., y_pred box): (K, (xmin, ymin, xmax, ymax))
         box_2(i.e., y_true box): (N, (xmin, ymin, xmax, ymax))

    return:
         iou: (k,), iou ratio
    '''

    #增加維度，用來利用broadcasting機制
    #box_1 shape -> (K, 1, 4)
    box_1 = tf.expand_dims(box_1, -2)
    #box_2 shape -> (1, N, 4)
    box_2 = tf.expand_dims(box_2, 0)

    #計算IOU
    intersect_mins = tf.maximum(box_1[...,0:2], box_2[...,0:2])
    intersect_maxes = tf.minimum(box_1[...,2:4], box_2[...,2:4])
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

    iou = intersect_area / (box_1_area + box_2_area - intersect_area)
    iou = tf.math.reduce_max(iou, axis=1)

    return iou

def box_diou(box_1, box_2):
    '''
    Args:
         box_1(i.e., y_pred box): (K, (xmin, ymin, xmax, ymax))
         box_2(i.e., y_true box): (N, (xmin, ymin, xmax, ymax))

    return:
         iou: (k,), iou ratio
    '''

    #增加維度，用來利用broadcasting機制
    #box_1 shape -> (K, 1, 4)
    box_1 = tf.expand_dims(box_1, -2)
    #box_2 shape -> (1, N, 4)
    box_2 = tf.expand_dims(box_2, 0)

    box_1_cen = tf.concat([(box_1[...,2:3] + box_1[...,0:1]) / 2,
                           (box_1[...,3:4] + box_1[...,1:2]) / 2],
                          axis=-1)

    box_2_cen = tf.concat([(box_2[...,2:3] + box_2[...,0:1]) / 2,
                           (box_2[...,3:4] + box_2[...,1:2]) / 2],
                          axis=-1)
    
    #兩個中心點的EU dis square
    rho_s = tf.math.reduce_sum(tf.math.square(box_1_cen - box_2_cen), -1)

    #對角線距離
    dia_mins = tf.minimum(box_1[...,0:2], box_2[...,0:2])
    dia_maxes = tf.maximum(box_1[...,2:4], box_2[...,2:4])

    regu = tf.math.divide_no_nan(rho_s, tf.math.reduce_sum(tf.math.square(dia_maxes - dia_mins), -1))

    #計算IOU
    intersect_mins = tf.maximum(box_1[...,0:2], box_2[...,0:2])
    intersect_maxes = tf.minimum(box_1[...,2:4], box_2[...,2:4])
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

    iou = intersect_area / (box_1_area + box_2_area - intersect_area)
    diou = iou - regu
    diou = tf.math.reduce_max(diou, axis=1)

    return tf.clip_by_value(diou, 0, 1)

def box_ciou(box_1, box_2):
    '''
    Args:
         box_1(i.e., y_pred box): (K, (xmin, ymin, xmax, ymax))
         box_2(i.e., y_true box): (N, (xmin, ymin, xmax, ymax))

    return:
         iou: (k,), iou ratio
    '''
    #增加維度，用來利用broadcasting機制
    #box_1 shape -> (K, 1, 4)
    box_1 = tf.expand_dims(box_1, -2)
    #box_2 shape -> (1, N, 4)
    box_2 = tf.expand_dims(box_2, 0)

    box_1_cen = tf.concat([(box_1[...,2:3] + box_1[...,0:1]) / 2,
                           (box_1[...,3:4] + box_1[...,1:2]) / 2],
                          axis=-1)

    box_2_cen = tf.concat([(box_2[...,2:3] + box_2[...,0:1]) / 2,
                           (box_2[...,3:4] + box_2[...,1:2]) / 2],
                          axis=-1)
    
    #兩個中心點的EU dis square
    rho_s = tf.math.reduce_sum(tf.math.square(box_1_cen - box_2_cen), -1)

    #對角線距離
    dia_mins = tf.minimum(box_1[...,0:2], box_2[...,0:2])
    dia_maxes = tf.maximum(box_1[...,2:4], box_2[...,2:4])

    regu = tf.math.divide_no_nan(rho_s, tf.math.reduce_sum(tf.math.square(dia_maxes - dia_mins), -1))

    box_1_wh = tf.concat([box_1[...,2:3] - box_1[...,0:1], box_1[...,3:4] - box_1[...,1:2]], -1)
    box_2_wh = tf.concat([box_2[...,2:3] - box_2[...,0:1], box_2[...,3:4] - box_2[...,1:2]], -1)

    box_1_ratio = tf.math.divide_no_nan(box_1_wh[...,0], box_1_wh[...,1])
    box_2_ratio = tf.math.divide_no_nan(box_2_wh[...,0], box_2_wh[...,1])
    
    #計算IOU
    intersect_mins = tf.maximum(box_1[...,0:2], box_2[...,0:2])
    intersect_maxes = tf.minimum(box_1[...,2:4], box_2[...,2:4])
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

    iou = intersect_area / (box_1_area + box_2_area - intersect_area)
    v = 4 / (math.pi**2) * tf.math.square(tf.math.atan(box_1_ratio) - tf.math.atan(box_2_ratio))
    alpha = tf.math.divide_no_nan(v ,(1 - iou) + v)
    ciou = iou - regu - alpha
    ciou = tf.math.reduce_max(ciou, axis=1)

    return tf.clip_by_value(ciou, 0, 1)

def preprocess_true_boxes(true_boxes, **kwargs):
    '''
    Args:
        true_boxes:(N, (xmin, ymin, xmax, ymax, cls index))
    '''
    true_boxes = true_boxes.astype(np.float32)
    #input shape(h, w)
    input_shape = np.array(kwargs['image_shape'], dtype='int32')
    #類別總數
    num_classes = kwargs['num_classes']
    anchors = np.array(kwargs['anchors'])
    anchors_mask = kwargs['anchors_mask']
    #每一層輸出的anchors數量
    anc_pre_l = len(anchors) // 3
    #w,h值
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    
    #3個輸出層的feature map大小
    grid_shapes = [input_shape // {0:8, 1:16, 2:32}[l] for l in range(3)]
    #初始化y_true
    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], anc_pre_l, 5+num_classes),
              dtype='float32') for l in range(3)]

    #增加anchors dimension，以便使用broadcasting
    anchors = np.expand_dims(anchors, 0)
    #ground truth boxes 與 anchors 算IOU，這邊看成所有box中心點為(0,0)算IoU。
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    #清除標記錯誤的data
    valid_mask_w = boxes_wh[...,0] > 0
    valid_mask_h = boxes_wh[...,1] > 0
    valid_mask = valid_mask_w * valid_mask_h
    boxes_wh = boxes_wh[valid_mask]
    true_boxes = true_boxes[valid_mask]

    #x,y中心點
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2
    boxes_xy = boxes_xy / input_shape[::-1]

    #增加dimension，方便使用broadcasting
    boxes_wh = np.expand_dims(boxes_wh, -2)
    box_maxes = boxes_wh / 2.
    box_mins = -box_maxes

    #計算IOU
    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    box_area = boxes_wh[...,0] * boxes_wh[...,1]
    anchor_area = anchors[...,0] * anchors[...,1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    #Multiple anchors for ground truth
    anchor_list = []
    for i in iou:
        anchor_list.append(np.argwhere(i > 0.213))
    #將label置入y_true中
    for box_ind, anchor in enumerate(anchor_list):
        #根據每個anchor放入對應值
        for i in anchor:
            #anchor 對應的輸出層
            out_level = int(i // 3)
            #anchor 對應的編號
            anchor_ind = int(i % anc_pre_l)
            i = np.floor(boxes_xy[box_ind, 0]*grid_shapes[out_level][1]).astype('int32')
            j = np.floor(boxes_xy[box_ind, 1]*grid_shapes[out_level][0]).astype('int32')
            c = true_boxes[box_ind, 4].astype('int32')
            y_true[out_level][j, i, anchor_ind, 0:4] = true_boxes[box_ind, 0:4]
            y_true[out_level][j, i, anchor_ind, 4] = 1
            y_true[out_level][j, i, anchor_ind, 5 + c] = 1
    
    #reshape對應到model輸出
    for i in range(3):
        y_true[i] = np.reshape(y_true[i], (-1, (num_classes + 5)))
    
    return np.vstack([y_true[0], y_true[1], y_true[2]])