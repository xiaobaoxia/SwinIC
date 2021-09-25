# SwinIC
Image Compression Using Swin Transformer
需要安装的包在文件requirement.txt中
训练集DIV2K 测试集Ko
首先使用utils将训练数据集预处理 需要配置数据集路径

运行train进行训练 命令如下

Command
两张卡训练网络 不加载预训练参数

python train.py --batchsize 16 --gpu "0,1" --gpu_count 2 --num_workers 16 --qp 7 --load_weights 0 --coco_root ../../../coco_dataset
以下是测试使用的训练参数

如果想要提升训练速度，需要增加GPU数量和batchsize，并且gpu_count需要同时修改为使用的GPU数量

通过QP参数加载不同的预训练参数和 lambda

num_workers是加载训练数据使用的进程数量，增加workers也可以在一定程度上增加训练速度，但是会消耗内存，CPU，GPU。所以num_workers不能太高，需要权衡

learning_rate 目前看来设置为默认值效果最好

训练的模型存放在train中
