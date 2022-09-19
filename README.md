# [Objects Matter: Learning Object Relation Graph for Robust Absolute Pose Regression](https://arxiv.org/abs/1909.03557)

Chengyu Qiao, Zhiyu Xiang, Xinglu Wang, Shuya Chen, Yuangang Fan and Xijun Zhao

## Introduction 

This is the PyTorch implementation of ORGMapNet, a neural architecture introducing object detections for robust visual localization.

## Setup

ORGMapNet uses a Conda environment that makes it easy to install all dependencies.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.

2. Create the `ORGMapNet` Conda environment: `conda env create -f environment.yml`.

3. Activate the environment: `conda activate orgmapnet_release`.

## Data
We support the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/), [RIO10](http://vmnavab26.in.tum.de/RIO10/) and [Oxford RobotCar](http://robotcar-dataset.robots.ox.ac.uk/) datasets right now. You can also write your own PyTorch dataloader for other datasets and put it in the `dataset_loaders` directory.

The datasets live in the `data/deepslam_data` directory.

### Special instructions for RobotCar:

1. Download [this fork](https://github.com/samarth-robo/robotcar-dataset-sdk/tree/master) of the dataset SDK, and run `cd data && ./robotcar_symlinks.sh` after editing the `ROBOTCAR_SDK_ROOT` variable in it appropriately.

2. For each sequence, you need to download the `stereo_centre`, `vo` and `gps` tar files from the dataset website. The directory for the scene has the .txt file defining the train/test_split.

3. To make training faster, we pre-processed the images using `scripts/process_robotcar.py`. This script undistorts the images using the camera models provided by the dataset, and scales them such that the shortest side is 256 pixels.

4. Pixel and Pose statistics must be calculated before any training. Use the `scripts/dataset_mean.py`, which also saves the information at the proper location. We provide pre-computed values for RobotCar and 7Scenes.

### 

## Running the code

### Training
The executable script is `train.py`. For example:

- ORGPosNet on `full` from `RobotCar`: 
```
python train.py --dataset RobotCar --scene full --config_file configs/orgposenet_RobotCar.ini \
--model orgposenet --device 0 --learn_beta --learn_gamma
```

- ORGMapNet on `full` from `RobotCar`: 
```
python train.py --dataset RobotCar --scene full --config_file configs/orgmapnet_RobotCar.ini \
--model orgmapnet --device 0 --learn_beta --learn_gamma
```

The meanings of various command-line parameters are documented in `train.py`.
The values of various hyperparameters are defined in a separate .ini file..

### Inference
The inference script is `scripts/eval.py`.
Here are some examples, assuming the models are downloaded in `scripts/logs`.

- ORGPoseNet on `full` from `RobotCar`: 
```
python eval.py --dataset RobotCar --scene full --model orgposenet --device 0 \
--weights logs/posenet/RobotCar/0/RobotCar_full_orgposenet_orgposenet_RobotCar_learn_beta_learn_gamma/epoch_100.pth.tar \
--config_file configs/orgposenet_RobotCar.ini --val
```

- ORGMapNet on `full` from `RobotCar`: 
```
python eval.py --dataset RobotCar --scene full --model orgmapnet --device 0 \
--weights logs/mapnet/RobotCar/0/RobotCar_full_orgmapnet_orgmapnet_RobotCar_learn_beta_learn_gamma/epoch_100.pth.tar \
--config_file configs/orgmapnet_RobotCar.ini --val
```

## Citation
If you find this code useful for your research, please cite our paper

```
@article{qiao4179862objects,
  title={Objects Matter: Learning Object Relation Graph for Robust Absolute Pose Regression},
  author={Qiao, Chengyu and Xiang, Zhiyu and Wang, Xinglu and Chen, Shuya and Fan, Yuangang and Zhao, Xijun},
  journal={Available at SSRN 4179862}
}
```

## Acknowledgements
Our code partially builds on [MapNet](https://github.com/NVlabs/geomapnet)

## License
Licensed under the MIT license, see [LICENSE](LICENSE.md).