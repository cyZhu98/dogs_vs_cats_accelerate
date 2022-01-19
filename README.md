# A PyTorch Example Using 🤗Accelerate Library in [Dogs vs. Cats Dataset.](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)

## TODO

- [ ] add test function

## Requirements

* pytorch (1.7.1)
* acclerate (0.5.0)
* timm (0.4.13)
* albumentations

最新版本也可以

## Run

**only support training with gpu**

make sure ```num_processes``` in [default_config.yaml](default_config.yaml) same as the number of you gpu

```bash
accelerate launch --config_file default_config.yaml run.py --DIR your_dataset_path
```

add ```--save``` in the end if you wish to save the best model

如果要保存最佳模型，在命令末尾加上```--save```

For more instruction about ```accelerater```, please refer to [its document](https://huggingface.co/docs/accelerate/quicktour.html) and its [example](https://github.com/huggingface/accelerate/blob/main/examples/cv_example.py).

## Explanation

This work is modified from [official PyTorch ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) and MoCo implementation.

model : pretrained swin transformer.

这个项目旨在用🤗acclerate库，取代原生的分布式训练所需要的繁琐的步骤。官方的ImageNet案例比较经典，就在其基础上做了修改。

#### 改动

猫狗数据集只有两类，所以删除了top5.

如果要把数据集换回ImageNet，只要按照官方案例把```getLoader```改成```datasets.ImageFolder```等等+修改模型的```num_classes```就可以了。

