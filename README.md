# A PyTorch Example Using 🤗Accelerate Library in [Dogs vs. Cats Dataset.](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)

## TODO

- [ ] add test function

## Requirements

* pytorch (1.7.1)
* acclerate (0.5.0)
* timm (0.4.13)

最新版本也可以

## Run

```bash
accelerate launch --config_file default_config.yaml run.py --DIR ..
```

add ```--save``` in the end if you wish to save the best model

如果要保存最佳模型，在命令末尾加上```--save```

For more instruction about ```accelerater```, please refer to [its document](https://huggingface.co/docs/accelerate/quicktour.html).

## Explanation

This work is modified from [official PyTorch ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) and MoCo implementation.