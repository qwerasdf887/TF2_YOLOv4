import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.regularizers import l2
import numpy as np
'''
YOLOv4 backboone的基礎元件
架構如下:
Conv2D -> Batch Normalization -> Mish
Conv2D -> Batch Normalization -> Leaky
Res Block
CSP Block
Neck Block (3.5層)
'''
class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
    
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

'''
tensorflow drop layer in training phase:
output = input / survival prob
survival prob = 1 - drop rate
i.e., 全1的輸入進入drop layer(假設drop rate = 0.4)
則輸出無丟棄的node值會是 1 / ( 1 - 0.4)
EfficientNet 的 drop_connect也是用此方式決定輸出值

但是DropBlock論文中，寫 A * count(M) / count_ones(M) 來當作輸出值
count(M): M的元素個數
count_ones(M): M中為1的個數

不確定是否採用算好的M來計算，但根據Tensorflow中的Drop與EfficientNet推斷，
採用第一種方式。
i.e., A = A / survival prob
'''
class DropBlock(tf.keras.layers.Layer):
    #drop機率、block size
    def __init__(self, drop_rate=0.2, block_size=3, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.rate = drop_rate
        self.block_size = block_size

    def call(self, inputs, training=None):
        if training:
            '''
            feature map mask tensor
            創建一個均勻取樣的Tensor，加上drop rate之後取整數，則為1的部份表示drop block的中心點
            經由maxpooling將中心點擴散成block size大小
            最後將Tensor值更改(0->1, 1->0)
            則可形成mask
            '''
            #batch size
            b = tf.shape(inputs)[0]
            
            random_tensor = tf.random.uniform(shape=[b, self.m_h, self.m_w, self.c]) + self.bernoulli_rate
            binary_tensor = tf.floor(random_tensor)
            binary_tensor = tf.pad(binary_tensor, [[0,0],
                                                   [self.block_size // 2, self.block_size // 2],
                                                   [self.block_size // 2, self.block_size // 2],
                                                   [0, 0]])
            binary_tensor = tf.nn.max_pool(binary_tensor,
                                           [1, self.block_size, self.block_size, 1],
                                           [1, 1, 1, 1],
                                           'SAME')
            binary_tensor = 1 - binary_tensor
            inputs = tf.math.divide(inputs, (1 - self.rate)) * binary_tensor
        return inputs
    
    def get_config(self):
        config = super(DropBlock, self).get_config()
        return config

    def build(self, input_shape):
        #feature map size (height, weight, channel)
        self.b, self.h, self.w, self.c = input_shape.as_list()
        #mask h, w
        self.m_h = self.h - (self.block_size // 2) * 2
        self.m_w = self.w - (self.block_size // 2) * 2
        self.bernoulli_rate = (self.rate * self.h * self.w) / (self.m_h * self.m_w * self.block_size**2)


def unit_conv(tensor, num_filters, k_size, strides, act, use_bn=True, drop_rate=0, block_size=3):
    #使用BN層，就不會使用conv的bias。
    #在YOLO中，padding與一般padding不同，只補一行一列
    padding = 'valid' if strides == 2 else 'same'
    if use_bn:
        #downsampling
        if strides == 2:
            tensor = tf.keras.layers.ZeroPadding2D(((1,0), (1,0)))(tensor)
        output = tf.keras.layers.Conv2D(filters=num_filters,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding,
                                        kernel_regularizer=l2(1e-5),
                                        use_bias=False)(tensor)
        output = tf.keras.layers.BatchNormalization()(output)
        if act == 'mish':
            output = Mish()(output)
        if act == 'leaky':
            output = tf.keras.layers.LeakyReLU(0.1)(output)

        output = DropBlock(drop_rate=drop_rate, block_size=block_size)(output)
        
    else:
        output = tf.keras.layers.Conv2D(filters=num_filters,
                                        kernel_size=k_size,
                                        padding=padding,
                                        kernel_regularizer=l2(1e-5))(tensor)
    return output

def Res_Block(tensor, num_filters, half=False, drop_rate=0, block_size=3):
    if half:
        x = unit_conv(tensor=tensor, num_filters=num_filters // 2, k_size=1, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    else:
        x = unit_conv(tensor=tensor, num_filters=num_filters, k_size=1, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    
    x = unit_conv(tensor=x, num_filters=num_filters, k_size=3, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    x = tf.keras.layers.Add()([x, tensor])
    return x

def Neck_Block(tensor, num_layers, num_filters, drop_rate=0, block_size=3):
    if num_layers == 3:
        x = unit_conv(tensor=tensor, num_filters=num_filters, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
        x = unit_conv(tensor=x, num_filters=num_filters * 2, k_size=3, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
        x = unit_conv(tensor=x, num_filters=num_filters, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    if num_layers == 5:
        x = unit_conv(tensor=tensor, num_filters=num_filters, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
        x = unit_conv(tensor=x, num_filters=num_filters * 2, k_size=3, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
        x = unit_conv(tensor=x, num_filters=num_filters, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
        x = unit_conv(tensor=x, num_filters=num_filters * 2, k_size=3, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
        x = unit_conv(tensor=x, num_filters=num_filters, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    return x

def CSP_Block(tensor, num_filters, num_repeat, half=True, drop_rate=0, block_size=3):
    #downsample:
    x = unit_conv(tensor=tensor, num_filters=num_filters, k_size=3, strides=2, act='mish', drop_rate=drop_rate, block_size=block_size)
    if half:
        num_filters = num_filters // 2
    #part 1
    part_1 = unit_conv(tensor=x, num_filters=num_filters, k_size=1, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    #part 2
    part_2 = unit_conv(tensor=x, num_filters=num_filters, k_size=1, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    #res block
    for i in range(0, num_repeat):
        part_2 = Res_Block(tensor=part_2, num_filters=num_filters, half=not(half), drop_rate=drop_rate, block_size=block_size)
    part_2 = unit_conv(tensor=part_2, num_filters=num_filters, k_size=1, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    part_2 = tf.keras.layers.Concatenate()([part_2, part_1])

    if half:
        output = unit_conv(tensor=part_2, num_filters=num_filters*2, k_size=1, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    else:
        output = unit_conv(tensor=part_2, num_filters=num_filters, k_size=1, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)

    return output

'''
將yolo輸出座標(xmin, ymin, xmax, ymax)、confidence、類別機率
input shape: (batch size, grid, grid, 255)
'''
class yolo_head(tf.keras.layers.Layer):
    def __init__(self, anchors, num_cls, img_shape):
        super(yolo_head, self).__init__()
        self.anchors = anchors
        self.num_anc = len(anchors)
        self.num_cls = num_cls
        self.scale_x_y = 1.1
        #shape = (h, w)
        self.shape = img_shape[:2]
    
    def call(self, inputs):
        #reshape layer to (batch size, gird, grid, anchors pre layer, (5+cls))
        x = tf.reshape(inputs, [-1, self.grid_shape[0], self.grid_shape[1], self.num_anc, (5+self.num_cls)])
        '''
        x_cen, y_cen = x[...,0:2]取sigmoid + cx, cy，再除以該feature map大小，獲得原圖的歸一化數據
        yolov4中引入"Eliminate grid sensitivity"，利用一個平移參數調整極端值問題
        '''
        box_xy = tf.sigmoid(x[...,0:2]) * self.scale_x_y - ((self.scale_x_y - 1) * 0.5) + self.grid
        #計算中心點於原始image的位置
        box_xy = (box_xy / tf.cast(self.grid_shape[::-1], tf.float32)) * tf.cast(self.shape[::-1], tf.float32)
        #w,h:取exp再乘以anchors得到實際寬高
        box_wh = tf.exp(x[...,2:4]) * self.anchors
        #計算(xmin, ymin, xmax, ymax)，並且限制範圍
        box_xmin, box_ymin = tf.split(box_xy - (box_wh / 2), num_or_size_splits=2, axis=-1)
        box_xmax, box_ymax = tf.split(box_xy + (box_wh / 2), num_or_size_splits=2, axis=-1)
        box_xmin = tf.clip_by_value(box_xmin, 0, self.shape[1])
        box_ymin = tf.clip_by_value(box_ymin, 0, self.shape[0])
        box_xmax = tf.clip_by_value(box_xmax, 0, self.shape[1] - 1)
        box_ymax = tf.clip_by_value(box_ymax, 0, self.shape[0] - 1)
        #形成(xmin, ymin, xmax, ymax)
        boxes = tf.concat([box_xmin, box_ymin, box_xmax, box_ymax], axis=-1)
        #confidence
        box_confidence = tf.sigmoid(x[..., 4:5])
        #classes
        box_class_prob = tf.sigmoid(x[..., 5:])

        output = tf.concat([boxes, box_confidence, box_class_prob], axis=-1)
        output = tf.reshape(output, [-1, self.grid_shape[0]*self.grid_shape[1]*self.num_anc, (5+self.num_cls)])
        return output
    
    def build(self, input_shape):
        #形成grid參數
        self.grid_shape = input_shape[1:3]
        self.grid = tf.meshgrid(tf.range(self.grid_shape[1]), tf.range(self.grid_shape[0]))
        self.grid = tf.expand_dims(tf.stack(self.grid, axis=-1), axis=2)
        self.grid = tf.cast(self.grid, tf.float32)


def creat_CSPYOLO(**kwargs):
#def creat_CSPYOLO(train_dict):

    #input shape(h, w)
    input_shape = (kwargs['image_shape'][0], kwargs['image_shape'][1], 3)
    #類別總數
    num_classes = kwargs['num_classes']
    #將anchors 歸一化
    anchors = np.array(kwargs['anchors'])# / kwargs['image_shape'][::-1]
    anchors_mask = kwargs['anchors_mask']
    #每一層輸出的anchors數量
    anc_pre_l = len(anchors) // 3
    #block drop參數
    drop_rate = kwargs['drop_rate']
    block_size = kwargs['block_size']

    input_layer = tf.keras.Input(shape=input_shape)
    ######CSPDarkBackbone#####
    x = unit_conv(tensor=input_layer, num_filters=32, k_size=3, strides=1, act='mish', drop_rate=drop_rate, block_size=block_size)
    block_1_f = CSP_Block(x, num_filters=64, num_repeat=1, half=False, drop_rate=drop_rate, block_size=block_size)
    block_2_f = CSP_Block(block_1_f, num_filters=128, num_repeat=2, drop_rate=drop_rate, block_size=block_size)
    block_3_f = CSP_Block(block_2_f, num_filters=256, num_repeat=8, drop_rate=drop_rate, block_size=block_size)
    block_4_f = CSP_Block(block_3_f, num_filters=512, num_repeat=8, drop_rate=drop_rate, block_size=block_size)
    block_5_f = CSP_Block(block_4_f, num_filters=1024, num_repeat=4, drop_rate=drop_rate, block_size=block_size)

    ######CSPDarkneck#####
    neck_5 = Neck_Block(tensor=block_5_f, num_layers=3, num_filters=512, drop_rate=drop_rate, block_size=block_size)
    #SPP
    max_1 = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding='same')(neck_5)
    max_2 = tf.keras.layers.MaxPool2D(pool_size=9, strides=1, padding='same')(neck_5)
    max_3 = tf.keras.layers.MaxPool2D(pool_size=13, strides=1, padding='same')(neck_5)
    neck_5 = tf.keras.layers.Concatenate()([max_3, max_2, max_1, neck_5])
    neck_5 = Neck_Block(tensor=neck_5, num_layers=3, num_filters=512, drop_rate=drop_rate, block_size=block_size)
    neck_5_upsample = unit_conv(tensor=neck_5, num_filters=256, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_5_upsample = tf.keras.layers.UpSampling2D()(neck_5_upsample)

    block_4_f = unit_conv(tensor=block_4_f, num_filters=256, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_4 = tf.keras.layers.Concatenate()([block_4_f, neck_5_upsample])

    neck_4 = Neck_Block(tensor=neck_4, num_layers=5, num_filters=256, drop_rate=drop_rate, block_size=block_size)
    neck_4_upsample = unit_conv(tensor=neck_4, num_filters=128, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_4_upsample = tf.keras.layers.UpSampling2D()(neck_4_upsample)

    block_3_f = unit_conv(tensor=block_3_f, num_filters=128, k_size=1, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_3 = tf.keras.layers.Concatenate()([block_3_f, neck_4_upsample])

    neck_3 = Neck_Block(tensor=neck_3, num_layers=5, num_filters=128, drop_rate=drop_rate, block_size=block_size)
    neck_3_out = unit_conv(tensor=neck_3, num_filters=256, k_size=3, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_3_out = unit_conv(tensor=neck_3_out, num_filters=((num_classes + 5) * anc_pre_l), k_size=1, strides=1, act='linear', use_bn=False)
    neck_3_out = yolo_head(anchors=anchors[anchors_mask[0]], num_cls=num_classes, img_shape=input_shape)(neck_3_out)
    
    neck_3_downsample = unit_conv(tensor=neck_3, num_filters=256, k_size=3, strides=2, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_4 = tf.keras.layers.Concatenate()([neck_3_downsample, neck_4])
    neck_4 = Neck_Block(tensor=neck_4, num_layers=5, num_filters=256, drop_rate=drop_rate, block_size=block_size)
    neck_4_out = unit_conv(tensor=neck_4, num_filters=512, k_size=3, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_4_out = unit_conv(tensor=neck_4_out, num_filters=((num_classes + 5) * anc_pre_l), k_size=1, strides=1, act='linear', use_bn=False)
    neck_4_out = yolo_head(anchors=anchors[anchors_mask[1]], num_cls=num_classes, img_shape=input_shape)(neck_4_out)

    neck_4_downsample = unit_conv(tensor=neck_4, num_filters=512, k_size=3, strides=2, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_5 = tf.keras.layers.Concatenate()([neck_4_downsample, neck_5])
    neck_5 = Neck_Block(tensor=neck_5, num_layers=5, num_filters=512, drop_rate=drop_rate, block_size=block_size)
    neck_5_out = unit_conv(tensor=neck_5, num_filters=1024, k_size=3, strides=1, act='leaky', drop_rate=drop_rate, block_size=block_size)
    neck_5_out = unit_conv(tensor=neck_5_out, num_filters=((num_classes + 5) * anc_pre_l), k_size=1, strides=1, act='linear', use_bn=False)
    neck_5_out = yolo_head(anchors=anchors[anchors_mask[2]], num_cls=num_classes, img_shape=input_shape)(neck_5_out)

    output = tf.keras.layers.Concatenate(axis=-2)([neck_3_out, neck_4_out, neck_5_out])

    #return tf.keras.Model(inputs=input_layer, outputs=[neck_3_out, neck_4_out, neck_5_out])
    return tf.keras.Model(inputs=input_layer, outputs=output)

def output_result(y_pred, **kwargs):
    '''
    Args:
        y_pred: model output (one image), list like: [[batch, total box, (xmin, ymin, xmax, ymax, cls_num)]]
    '''
    print('score threshold:', kwargs['score_threshold'])
    num_box = []
    num_cls = []
    num_score = []
    for pred_out in y_pred:
        pred_out = tf.reshape(pred_out, [-1, 5 + kwargs['num_classes']])
        boxes = tf.concat([pred_out[...,1:2],
                           pred_out[...,0:1],
                           pred_out[...,3:4],
                           pred_out[...,2:3]], axis=-1)
        box_conf = pred_out[...,4:5]
        box_cls = pred_out[...,5:]
        mask = tf.reshape((box_conf > kwargs['score_threshold']), (-1,))
        box_conf = tf.boolean_mask(box_conf, mask)
        box_cls = tf.boolean_mask(box_cls, mask)
        boxes = tf.boolean_mask(boxes, mask)

        num_box.extend(boxes)
        num_cls.extend(box_cls)
        num_score.extend(box_conf)
    
    num_box = np.array(num_box)
    num_cls = np.array(num_cls)
    num_score = np.array(num_score)

    if len(num_box) == 0:
        return num_box, num_cls, num_score
    else:
        #暫時使用普通NMS
        nms_index = tf.image.non_max_suppression(num_box, tf.reshape(num_score, (-1,)), 40, iou_threshold=kwargs['iou_threshold'])
        select_boxes = tf.gather(num_box, nms_index)
        select_cls = tf.math.argmax(tf.gather(num_cls, nms_index), -1)
        select_score = tf.gather(num_score, nms_index)

        return select_boxes, select_cls, select_score

'''
load weights:
reference: https://github.com/hunglc007/tensorflow-yolov4-tflite
'''
def load_weights(model, weights_file, custom_cls=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    
    j = 0
    for i in range(110):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        #output layer
        if i not in [93, 101, 109]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            filters = 255
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [93, 101, 109]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            if not(custom_cls):
                conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    print("load OK")
    wf.close()

#等比例縮放影像
def resiresize_img(image, box_loc=None, **kwargs):
    h, w, _ = image.shape
    max_edge = max(608, 608)
    scale = min( max_edge / h, max_edge / w)
    h = int(h * scale)
    w = int(w * scale)

    if box_loc is None:
        return cv2.resize(image, (w, h))
    else:
        box_loc[:,0] = box_loc[:,0] * scale
        box_loc[:,1] = box_loc[:,1] * scale
        box_loc[:,2] = box_loc[:,2] * scale
        box_loc[:,3] = box_loc[:,3] * scale
        return cv2.resize(image, (w, h)), box_loc.astype(np.int32)

#將樸片補至指定大小
def padding_img(image, box_loc=None, **kwargs):
    h, w, _ = image.shape

    dx = int((608 - w) / 2)
    dy = int((608 - h) / 2)

    out_img = np.ones((608, 608, 3), np.uint8) * 127
    out_img[dy: dy + h, dx: dx + w, :] = image

    if box_loc is None:
        return out_img
    else:
        box_loc[:,0] = box_loc[:,0] + dx
        box_loc[:,1] = box_loc[:,1] + dy
        box_loc[:,2] = box_loc[:,2] + dx
        box_loc[:,3] = box_loc[:,3] + dy
        return out_img, box_loc.astype(np.int32)

def draw_img(img, boxes, cls, socre, **kwargs):
    h, w = kwargs['image_shape']
    #boxes = np.clip(boxes, 0, 1)
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 255), 2)
        print(int(box[1]), int(box[0]), int(box[3]), int(box[2]))

if __name__ == "__main__":
    import cv2
    import time
    default = {'anchors': [[12, 16], [19, 36], [40, 28],
                           [36, 75], [76, 55], [72, 146],
                           [142, 110], [192, 243], [459, 401]],
               'anchors_mask': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
               'image_shape': (608, 608),
               'num_classes': 80,
               'score_threshold': 0.6,
               'iou_threshold': 0.3,
               'batch_size': 2,
               'drop_rate' : 0.2,
               'block_size' : 3
              }

    model = creat_CSPYOLO(**default)
    model.summary()
    start_1 = time.time()
    load_weights(model, '../yolov4.weights')
    print('load weight times:', time.time() - start_1)
    img = cv2.imread('../kite.jpg')
    img = resiresize_img(img)
    img = padding_img(img)
    pred_img = np.expand_dims(img, axis=0).astype(np.float32) / 255
    start_2 = time.time()
    result = model.predict(pred_img)
    boxes, class_, score = output_result(result, **default)
    print('predict times:', time.time() - start_2)
    draw_img(img, boxes, class_, score, **default)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()