# A PyTorch Example Using 🤗Accelerate Library in [Dogs vs. Cats Dataset.](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)

## TODO

- [x] add test function

## Overview

Public score: 0.03304

#### 耗时对比

Accelerate : **1043.63** *seconds*

Data Parallel : **1055.52** *seconds*

## Requirements

* pytorch (1.7.1)
* acclerate (0.5.0)
* timm (0.4.13)
* albumentations

最新版本也可以

## Run

### Train

**only support training with gpu**

1. download [Dataset.](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)

2. make sure ```num_processes``` in [default_config.yaml](default_config.yaml) same as the number of you gpu

```bash
accelerate launch --config_file default_config.yaml run.py --DIR your_dataset_path
```

add ```--save``` in the end if you wish to save the best model

如果要保存最佳模型，在命令末尾加上```--save```

For more instruction about ```accelerater```, please refer to [its document](https://huggingface.co/docs/accelerate/quicktour.html) and its [example](https://github.com/huggingface/accelerate/blob/main/examples/cv_example.py).

(**Optional**) Dataparallel : ```python run_dataparallel.py -- DIR your_dataset_path```

### Test

```bash
python test.py --DIR your_dataset_path --checkpoint model_save_path
```

example:  ```python test.py --DIR .. --checkpoitn ep1_acc99453```

## Explanation

This work is modified from [official PyTorch ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) and MoCo implementation.

model : pretrained swin transformer.

这个项目旨在用🤗acclerate库，取代原生的分布式训练所需要的繁琐的步骤。官方的ImageNet案例比较经典，就在其基础上做了修改。

#### 改动

猫狗数据集只有两类，所以删除了top5.

如果要把数据集换回ImageNet，只要按照官方案例把```getLoader```改成```datasets.ImageFolder```等等+修改模型的```num_classes```就可以了。

我看案例中没有指定```args.rank==0```才能```print```，所以运行的话应该会，有几张卡就会有几行重复的输出信息。我的代码中所有的```print```都替换为```accelerator.print```，作用是只有进程在gpu0 (rank=0)的时候才会输出信息(等于```if args.rank==0: print()```

#### 说明

按照个人的风格，将代码拆分成了几个模块。

如果要进一步刷分，可以增加更多的数据增强策略，更改模型等等。