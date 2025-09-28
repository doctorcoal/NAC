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
## Conducting NAC on pre-trained models
Choose from following commands to apply NAC to our pretrained models.

To compress ResNet34 models with NAC:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model resnet_exit_quant --dataset cifar10 --arinc --suffix t1
CUDA_VISIBLE_DEVICES=0 python main.py --model resnet_exit_quant --dataset cifar100 --arinc --suffix t1
CUDA_VISIBLE_DEVICES=0 python main.py --model resnet_exit_quant --dataset tiny-imagenet --arinc --suffix t1
```

To compress MobileNetV2 models with NAC:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model mobilenetV2_hira_quant --dataset cifar10 --arinc --suffix t1
CUDA_VISIBLE_DEVICES=0 python main.py --model mobilenetV2_hira_quant --dataset cifar100 --arinc --suffix t1
CUDA_VISIBLE_DEVICES=0 python main.py --model mobilenetV2_hira_quant --dataset tiny-imagenet --arinc --suffix t1
```

To compress DeiT-tiny models with NAC:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model DeiT-tiny --dataset cifar10 --arinc --suffix t1
CUDA_VISIBLE_DEVICES=0 python main.py --model DeiT-tiny --dataset cifar100 --arinc --suffix t1
CUDA_VISIBLE_DEVICES=0 python main.py --model DeiT-tiny --dataset tiny-imagenet --arinc --suffix t1
```

To apply a fixed compression strategy on pre-trained models with CST (compression strategy transfering):
```
CUDA_VISIBLE_DEVICES=0 python main.py --model [model_name] --dataset [dataset_name] --arinc --suffix CST --regen_traj [compression strategy]
```

To apply post-finetuning on NAC/CST compressed models:
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --model [model_name] --dataset [dataset_name] --env_file_name [compression strategy] --suffix post
```

### Config and Hyperparameters
You can reconfigure almost all hyperparameters in `./env/data.py`.

### Results

Optimal compression strategies obtained by Compression Space Search with post-finetuning.


| Model       | Dataset       | Trajectory       | Acc.    | Acc. loss| BitOpsCR| MemCR  | GPU hours  |
|-------------|---------------|------------------|---------|----------|---------|--------|------------|
| ResNet34    | cifar10       | QQFEEEEEEEEP     | 90.88   | 1.30     | 100.80  | 28.56  | 29.00      |
|             | cifar100      | PQQFE            | 78.07   | 1.01     | 35.50   | 8.23   | 36.00      |
|             | tiny-imagenet | QPPPFPPPQFPEPPF  | 65.35   | 1.85     | 37.87   | 10.19  | 140.00     |
| MobileNetV2 | cifar10       | QPPPFPPFEPPQFPPF | 93.35   | 0.75     | 47.48   | 10.85  | 21.00      |
|             | cifar100      | EQQFPF           | 76.82   | 0.89     | 35.93   | 8.17   | 32.00      |
|             | tiny-imagenet | PQPPFQFPFPFEE    | 62.78   | 1.98     | 41.08   | 9.02   | 79.00      |
| DeiT-tiny   | cifar10       | QPPPPPPPPQPPF    | 90.80   | 0.10     | 41.34   | 10.06  | 18.00      |
|             | cifar100      | QQEPPF           | 69.00   | 1.60     | 34.19   | 8.37   | 25.00      |
|             | tiny-imagenet | QEPQF            | 56.50   | -1.10    | 35.06   | 8.18   | 46.00      |

## Custom model compression
To compress your own model with NAC, you need to make sure your model is compatible with the compression pipeline.
This requires minimal changes to your original model code. Below we describe the general steps, using MobileViT as an example.
1. Inherit from Comp_Base class
Your model class should inherit from Comp_Base instead of plain nn.Module.
This allows the NAC tool to register forward hooks, track blocks, and manage exits.
```
class MobileViT(Comp_Base):   # Inherit Comp_Base
    def __init__(..., 
                 exit_type='LG-deit-8', exit_mode='original', w=32, a=32,
                 place_layer={1: 'mlp', 3: 'mlp'}):
        super().__init__(num_classes=num_classes,
                         depth=1,             # for exit depth control
                         w=w, a=a,            # bitwidth for quantization
                         exit_mode=exit_mode, # exit policy
                         exit_type=exit_type, # exit type (e.g. LG-deit-8)
                         place_layer=place_layer)
```
2. Initialize exits
Call the helper provided by Comp_Base to automatically insert early-exit heads:
```
super().init_exit(
    self,
    exit_type=exit_type,
    channels=channels,
    num_classes=num_classes,
    image_size=image_size,
    place_layer=place_layer
)
```
3. Maintain a blocklist
Your blocks should be stored in self.blocklist rather than arbitrary names.
```
self.blocklist = nn.ModuleList([])
```
4. Implement block_forward
This function defines how a single block is executed and profiled. Modify it based on your model.
```
def block_forward(self, x, i, block):
    self.inf_start()
    x = block(x)
    self.inf_record()
    return x
```
5. Implement exit_forward
This function defines how an exit head is executed.
```
def exit_forward(self, x, i, block):
    self.exit_start()
    exit_x = self.exit_heads[i](x)
    self.exit_record()
    return exit_x
```
6. Full Example
Here is a minimal version of a custom model ready for compression with NAC:
```
class MyModel(Comp_Base):
    def __init__(self, num_classes, w=32, a=32,
                 exit_type='LG-deit-8', exit_mode='original',
                 place_layer={1: 'mlp'}):
        super().__init__(num_classes=num_classes, depth=10,
                         w=w, a=a, exit_mode=exit_mode,
                         exit_type=exit_type, place_layer=place_layer)

        # define layers
        self.block1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.block2 = nn.Linear(16, num_classes)

        # register exits
        super().init_exit_mobilevit(self, exit_type=exit_type,
                                    channels=[16, num_classes],
                                    num_classes=num_classes,
                                    image_size=(32, 32),
                                    place_layer=place_layer)

        # blocklist for compression
        self.blocklist = nn.ModuleList([self.block1, self.block2])

    def block_forward(self, x, i, block):
        self.inf_start()
        x = block(x)
        self.inf_record()
        return x

    def exit_forward(self, x, i, block):
        self.exit_start()
        exit_x = self.exit_heads[i](x)
        self.exit_record()
        return exit_x
```
