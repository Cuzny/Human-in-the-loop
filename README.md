# Human-in-the-loop

### Abstract

Image classification task using the human-in-the-loop. During the training stage, the feature extraction network is fixed while the classification network is iteratively updated according to the formula based on paper: Human-In-The-Loop Person Re-Identification[https://arxiv.org/abs/1612.01345]



### Requirements

- Python (3.7)
- Pytorch (1.7.0)
- PyQt5 & PyQt5-tools （https://blog.csdn.net/zyngoo/article/details/85880572）(use pip install)

**datasets:** CIFAR100. The datasets can be downloaded by following the steps listed at https://www.robots.ox.ac.uk/~vgg/decathlon/#download. 

**models**: Feature extraction network: resnet18, pretrained on 90 classes of CIFAR100 ()

Classification network: , trained on the last 10 classes of CIFAR100 (Human-in-the-loop)



### Training models


To select by human:

```
$ python train.py --eval_time 100 --train_batch 2 --val_batch 2 --train_class_num 10 --gallery_size 10 --max_epoch 6 --is_simu 0
```

