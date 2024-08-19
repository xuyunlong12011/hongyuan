
# import sys
# sys.path.append(r"D:\Pycharm-Projects\ultralytics\ultralytics")#直接强制扫描ultralytics，不会出现No module named 'ultralytics'，一定要放在最上面


# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# if __name__ == '__main__':
#     model = YOLO(r'D:\Pycharm-Projects\ultralytics\ultralytics\cfg\models\v8\yolov8n.yaml')

#     model.load(r'D:\Pycharm-Projects\ultralytics\yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度

#     model.train(
#                 data=r'D:\Pycharm-Projects\ultralytics\ultralytics\cfg\datasets\AI-DOT.yaml',  #  将data后面替换你自己的数据集地址
#                 cache=True,#如果设置为True，在训练前将整个数据集加载到内存中，可以加速训练过程，但需要较多内存
#                 imgsz=640,#训练时输入图像的大小，这里设置为640x640像素。图像尺寸对模型性能和速度有重要影响
#                 epochs=30,#训练的总轮数，这里设置为100轮。一个epoch表示整个数据集被模型看过一次
#                 single_cls=False,  # 是否是单类别检测,如果设置为True，表示模型仅需要识别一种类别。
#                 batch=4,#批大小，每次迭代训练所用的样本数量。这里设置为1，通常较小的批大小会使训练过程更稳定，但可能会增加训练时间
#                 close_mosaic=10,#是一个特定于YOLOv8的参数，用于控制何时停止使用mosaic数据增强（一种图像数据增强方法）
#                 workers=2,#用于加载数据的工作线程数。设置为0表示数据加载将在主线程中进行。
#                 device='0',#指定训练使用的设备，'0'通常表示使用第一个GPU
#                 optimizer='auto',  # using SGD,选择优化器，这里使用SGD（随机梯度下降）
#                 amp=True,  # 自动混合精度训练，可以提高训练速度和效率，同时减少内存使用。如果训练中出现损失为NaN，可以关闭此功能
#                 project='runs/train',
#                 name='v8-AI_DOT',#训练实验的名称，输出文件将保存在project/name目录下。
#                 )




#windows下面的断点续训
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"D:\Pycharm-Projects\ultralytics\runs\train\v8-AI_DOT\weights\last.pt")
    # 中断训练的权重文件中的last.pt
    results = model.train(resume=True)