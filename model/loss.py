import tensorflow as tf
from .utils import box_ciou
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

def yolo_loss(y_true, y_pred):
    '''
    Args:
        y_true:(batch size, gird*grid*num_anchors, (xmin, ymin, xmax, ymax, conf, cls))
        y_pred:(batch size, gird*grid*num_anchors, (xmin, ymin, xmax, ymax, conf, cls)
    '''
    batch_size = tf.shape(y_pred)[0]
    def loop_loss(index, sum_loss):
        #pred box (box, (xmin, ymin, xmax, ymax))
        pred_box = y_pred[index,:,0:4]
        #找出標記ground truth 的anchor
        obj_mask = tf.reshape(y_true[index,:,4:5], (-1,))
        boxes = tf.boolean_mask(y_true[index,:,0:4], obj_mask)
        #使用nms還原ground truth box
        box_ind = tf.image.non_max_suppression(boxes, tf.ones(tf.shape(boxes)[0], tf.float32), 40)
        boxes = tf.gather(boxes, box_ind)

        #計算所有預測bbox 與 ground truth bbox 的 CIoU
        CIoU = box_ciou(y_pred[index,:,0:4], boxes)
        CIoU_loss = tf.reduce_sum(tf.boolean_mask((1 - CIoU), obj_mask))

        #classes loss (只計算正樣本)
        cls_loss = categorical_crossentropy(y_true[index,:,5:], y_pred[index,:,5:])
        cls_loss = tf.reduce_sum(tf.boolean_mask(cls_loss, obj_mask))

        #計算其他預測bbox與ground truth bbox IoU，如果大於ignore_thresh則不計算conf loss
        ignore_mask = CIoU < 0.7
        #利用ignore_mask與obj_mask算出參與計算的conf loss
        mask = tf.math.logical_or(ignore_mask, tf.cast(obj_mask, tf.bool))
        #confidience loss
        conf_loss = binary_crossentropy(y_true[index,:,4:5], y_pred[index,:,4:5])
        conf_loss = tf.reduce_sum(tf.boolean_mask(conf_loss, mask))
        return index+1, sum_loss + CIoU_loss + cls_loss + conf_loss
    
    _, final_loss = tf.while_loop(lambda index, sum_loss: index < batch_size, loop_loss, [0, 0.])

    return final_loss / tf.cast(batch_size, tf.float32)