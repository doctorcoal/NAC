# NAC: Neural Network Automatic Chain Compressor
![framework overview](framework%20overview.png "NAC Framework Overview")
## Requirements
For standard use of NAC, please prepare the required environment, datasets and models.
### Environment
Use the following code to start and complete the configuration of the virtual environment:

```setup
conda create -n NAC python=3.12.3
conda activate NAC
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
### Pre-trained Models
Please download our pretrained models here for Neural Network Automatic Chain Compressor (NAC):

- [pre-trained model with exit layers](https://1drv.ms/u/c/9230e2f0a40a705d/ES0Jne_HfNJCmaDvHNdZmkwBNwo7cXOmemrGaK8vThuJfg?e=GBgtz3) trained on Cifar10, Cifar100,  Tiny-ImageNet and ImageNet-1k.

- Download `.zip` file into `./result/` folder and unzip it.

### Dataset
Firstly change the dataset location in `./data.py`.
```
_DATASETS_MAIN_PATH = '[place to download dataset]'
```

Then download those datasets.
Cifar10 and Cifar100 can be automatically downloaded. For ImageNet, refer to [Download, pre-process, and upload the ImageNet dataset](https://cloud.google.com/tpu/docs/imagenet-setup).
For Tiny-ImageNet, [download](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it, then run following python script to correct its file structure.
```
import glob
import os
from shutil import move
from os import rmdir

target_folder = './val/'

val_dict = {}
with open('./val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]
        
paths = glob.glob('./val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
       
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)
    
rmdir('./val/images')
```
## Quick Start
After everything has been installed, you can use the following command to start compressing pretrained ResNet34 model on cifar10 with NAC to verify if the environment is configured properly.

```
CUDA_VISIBLE_DEVICES=0 python main.py --model resnet_exit_quant --dataset cifar10 --arinc --suffix t1
```
