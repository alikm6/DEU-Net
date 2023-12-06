# DEU-Net: Dual-Encoder U-Net for Automated Skin Lesion Segmentation

This repository contains the official PyTorch implementation of "[DEU-Net: Dual-Encoder U-Net for Automated Skin Lesion Segmentation](https://ieeexplore.ieee.org/document/10332179)".

![DEU-Net architecture](images%2FDEU-Net.png)

## Prerequisites
- PyTorch 2.0.0
- Python 3.9
- Additional packages listed in requirements.txt.

## Data Preparation
We use four datasets in this project: ISIC2016, ISIC2017, ISIC2018, and PH2. 

1. Download the ISIC datasets from https://challenge.isic-archive.com/data/ and place them into the corresponding folders: datasets/ISIC2016, datasets/ISIC2017, and datasets/ISIC2018.

2. Download the PH2 dataset from https://www.fc.up.pt/addi/ph2%20database.html and place it in the datasets/PH2 folder.

Once you have downloaded the data, you need to preprocess it using the following steps:

1. (Optional) To increase network training speed, you can use the `preprocess_resize_dataset.py` script to resize the dataset to your desired dimensions. Note that this step is not necessary, as the network training script will automatically resize the input data.

2. Use the `preprocess_split_dataset.py` script to specify training, validation, and test data and store them in the data folder. The files generated will be used for training and testing the network.

## Network Training
You can train the network using the `train.py` script, which takes the following arguments:

- `--net-cfg`: A configuration file for the desired network (examples can be found in cfg/net folder)
- `--train-cfg`: A configuration file for the network training settings, such as the number of epochs, optimizer, etc. (example can be found at cfg/train.cfg)
- `--dataset`: The directory of the dataset (this is the same folder obtained from the second stage of data preparation)

There are several other arguments that can be found in the code.

## Network Evaluation
You can evaluate the network using the `evaluate.py` script, which takes the following arguments:

- `--cfg`: One or more config files for the desired networks
- `--model`: One or more files containing the weights of the networks. Multiple models can be used simultaneously for an ensemble output.
- `--transforms`: A list of transformations to perform on each input image and calculate the final result from the combination of results (possible transformations: vflip, hflip, rotation_90, rotation_180, rotation_270)
- `--input-images`: A txt file containing the paths of the images to be segmented
- `--input-masks`: A txt file containing the paths of the ground truth masks corresponding to the input images

If you don't want to apply any transformations and only want to segment the original image, leave the `--transforms` argument empty.

There are several other arguments that can be found in the code.

## Prediction
You can use the `predict.py` script to produce output for multiple images. The script takes the following arguments:

- `--cfg` and `--model` and `--transforms`: Similar to `evaluate.py`. Multiple models can be used simultaneously for an ensemble output.
- `--input`: A list of paths for input images to be segmented

There are several other arguments that can be found in the code.

## Citation

If you find this repo helpful, please consider citing our paper:

```
@ARTICLE{10332179,
  author={Karimi, Ali and Faez, Karim and Nazari, Soheila},
  journal={IEEE Access}, 
  title={DEU-Net: Dual-Encoder U-Net for Automated Skin Lesion Segmentation}, 
  year={2023},
  volume={11},
  number={},
  pages={134804-134821},
  doi={10.1109/ACCESS.2023.3337528}}
```