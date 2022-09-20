# [Objects Matter: Learning Object Relation Graph for Robust Absolute Pose Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4179862)

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

### Object Detections
Objects of interest are obtained by the object detector [FCOS](https://github.com/tianzhi0549/FCOS) and saved as .npz files.

As a dictionary, the .npz file needs to contain three items:

1. bbox : numpy array, the sizes and positions of the object boxes.

2. idx : numpy array, the serial numbers of the object categoriers.

3. label_nm : string, the classes of objects.

We adopt different strategies for different datasets to obtain the groundtruth object boxes for training.

For the 7Scenes dataset, we directly use the FCOS network trained on the COCO dataset to obtain object detections.

For the RIO10 dataset, we obtain the groundtruth object boxes based on the provided instance segmentation labels.

For the RobotCar dataset, we annotate static objects in a small number of images as groundtruth boxes.

For training FCOS, please first add the image and label file paths to `FCOS/fcos_core/config/paths_catalog.py`,
then modify the .yaml configuration file in the `FCOS/configs` according to the dataset.
The [model](https://drive.google.com/drive/folders/1QHToZChZSddiLJ5DEyoM6nQ3bG29ANXn?usp=sharing) trained on the RobotCar dataset can be download here.

Besides, you can leverage various robust object detection frameworks in the widely used [MMDetection](https://github.com/open-mmlab/mmdetection) to generate object detections.
We provide the relevant configuration file and the [model](https://drive.google.com/drive/folders/1rytwsakcVdEoG-eNgl5i09_B05O2GCIl?usp=sharing) trained on the RobotCar dataset to make it easier for you to use MMDetection.

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