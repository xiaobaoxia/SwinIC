# SwinIC
Image Compression Using Swin Transformer
项目依赖在文件requirement.txt中

训练集DIV2K train&val 解压到dataset/DIV2K/DIV2K_train_HR中。
下载链接 
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip 
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

测试集Kodak 已在dataset/kodak

首先使用utils.py将训练数据集进行预处理
900张图上采样到(2000,2000) 随机裁剪384 

请在train.py 中配置真实数据集路径
```python
parser.add_argument(
        "--train_path", default='dataset/DIV2K/DIV2K_train_HR_sub', type=str,
        help='train dataset path')
parser.add_argument(
        "--val_path", default='dataset/kodak', type=str,
        help='val dataset path')
```

运行train进行训练 命令如下

Command
需要完成以下11组参数的训练 
batchsize和使用的gpu数量需要自行调整
batchsize为4时约需要11G显存
```shell
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 0 --qp 1
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 0 --qp 2
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 0 --qp 3
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 0 --qp 4
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 0 --qp 5
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 0 --qp 6
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 0 --qp 7
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 1 --qp 1
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 1 --qp 2
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 1 --qp 3
python train.py --batchsize 32 --gpu "0,1,2,3" --model_type 1 --qp 4
```

model_type 0 损失为MSE 
model_type 0 损失为MS-SSIM

