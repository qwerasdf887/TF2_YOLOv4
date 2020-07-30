# TF2_YOLOv4  
Tensorflow 2.2版本的YOLOv4，幾乎個function都有相對應的註解。  
若有任何問題歡迎討論，Model layer加入Drop Block layer。  
Loss function : CIOU

## 環境 environment 

1. Tensorflow 2.2
2. tensorflow_addons (moving average opt)
3. Python 3.5~3.7
4. OpenCV 3~4

## Tiny version

[Tiny-YOLOv4](https://github.com/qwerasdf887/TF2_TinyYOLOv4)

## Dropblock  
`train_default`下的`dropblock: [False, True, True, True, True]`分別對應每個大的residual block後面增加。  

## Weights

ALexeyAB : [download weight](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

## YOLOv4架構如下

<p align="center">
    <img width="100%" src="https://github.com/qwerasdf887/TF2_YOLOv4/blob/master/YOLOv4.png" style="max-width:100%;">
    </a>
</p>


## Predict Img:

```bashrc
python predict.py
```

### Result

<p align="center">
    <img width="100%" src="https://github.com/qwerasdf887/TF2_YOLOv4/blob/master/predict.jpg" style="max-width:100%;">
    </a>
</p>


## Training

需修改train.py。
使用[labelImg](https://github.com/tzutalin/labelImg) 生成的xml檔，並且放入標籤位置即可。
