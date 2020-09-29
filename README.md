# Introduction

This repository is a PyTorch implementation of [DenseFuse: A Fusion Approach to Infrared and Visible Images](https://ieeexplore.ieee.org/document/8580578). You can see the original code in [here](https://github.com/hli1221/imagefusion_densefuse). I made their codes adaptive for the latest version of PyTorch(=1.6.0).

# Features

This repository has several features as follows

* PyTorch implementation
* Multi-GPU training
* Support training/testing both 1-channel and 3-channel model
* You can run demo with MS COCO-based pretrained model (.pt file)

# Prerequisites

* PyTorch
* tqdm
* pillow
* numpy

# How to train

* It is enough to download [MS COCO dataset](https://cocodataset.org/#home) for training.
* Set your training phase at args_fusion.py
* For multi-GPU training, you should set parameters as follows,
```
### args_fusion.py
# For GPU training
world_size = -1
rank = -1
dist_backend = 'nccl'
gpu = 0,1,2,3
multiprocessing_distributed = True
distributed = None
```
You can see details of these parameters at [tutorials of PyTorch official documents](https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training). 
* If you have pretrained models, you can transfer the training to them.
```
### args_fusion.py
# resume = "models/rgb.pt" # Transfer learning
resume = None # Train from scratch
```
* Make train.txt file which contains paths of training datas. For example,
```
/home/kim/images/1.jpg
/home/kim/images/2.jpg
/home/kim/images/3.jpg
```
* Run the command below
```
python train.py
```

# How to test

* Please add the path of datasets for test.
```
### args_fusion.py
strategy_type = "attention" # addition or attention
test_save_dir = "./"
test_img = "./test_rgb.txt"
test_ir = "./test_ir.txt"
```
* For testing, I recommand [KAIST Multispectral Pedestrian Detection Benchmark](https://soonminhwang.github.io/rgbt-ped-detection/).

# To-Do list
* [ ] Make a demo for video sequences
* [ ] Performance benchmark

# Citation

 *H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May 2019.*

```
@article{li2018densefuse,
  title={DenseFuse: A Fusion Approach to Infrared and Visible Images},
  author={Li, Hui and Wu, Xiao-Jun},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={5},
  pages={2614--2623},
  month={May},
  year={2019},
  publisher={IEEE}
}
```




