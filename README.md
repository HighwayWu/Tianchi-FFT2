# [Tianchi Competition "Forgeries and Forensics" Track 2](https://tianchi.aliyun.com/competition/entrance/531812/introduction)

An official implementation code of Rank 3.

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Demo](#demo)


## Background

<p align='center'>  
  <img src='https://github.com/HighwayWu/Tianchi-FFT2/blob/master/imgs/demo.png' width='1060'/>
  Demo. The upper part shows the inputs (forged images) and the bottom part is the corresponding outputs (detection results).
</p>

In this competition, the competitors are required to proposed an algorithm for detect and locate the forged regions from an input image. The classic forensics algorithms are to locate the forged area by studying the anomalous clues in the image, e.g., JPEG compression artifacts and/or noise pattern. However, these cues are not robust enough, as almost all the input images undergo multiple complex post-processing, making the classic forensics unable to accurately detect and locate the forged regions. With the rapid development of deep learning (DL) in recent years, many DL-based models have been proposed for image forensics as their strong learning representation and generalization abilities.

Thereby, we proposed to use the U-Net [1] that utilizes Se-Resnext50 [2] as encoder and incorporates SCSE [3] attention module. It is worth mentioning that the SCSE attention module is effective as the "attention" operation performed by it can well allow the model to re-weight the characteristics of the tampered area. After proper data enhancement and validation set partitioning, we trained four models for model ensemble.

[1] Ronneberger. et. al., "U-net: Convolutional networks for biomedical image segmentation." [Link.](https://arxiv.org/abs/1505.04597)

[2] Hu et. al., "Squeeze-and-excitation networks." [Link.](https://arxiv.org/abs/1709.01507)

[3] Roy et. al., "Recalibrating fully convolutional networks with spatial and channel "squeeze and excitation" blocks." [Link.](https://arxiv.org/abs/1808.08127)

## Dependency
Please refer to the "requirements.txt" file.

## Demo

To train the model:
```bash
sh code/train.sh
```
Note: the training/testing data can be download from the official website.

To test the model:
```bash
sh code/run.sh
```
Then the model will detect the images in the `../s2_data/data/test/` and save the results in the `../prediction_result/images/` directory.

The pre-trained model weights are avaliable in [here](https://drive.google.com/file/d/1ork_zGlG-Ny5sOLPOYK7RIppZKwOCm40/view?usp=sharing). Please download them and put in the "../user_data/model_data/".

More explanation for "code/run.sh":

    For "python main.py test --func=0": the "func=0" means the division of input images. The divided sub-images are save in "../s2_data/data/test_decompose_\*" (\* is the resolution of divided sub-images).
		
    For "python main.py test --func=1 --size_idx=0 --fold=1 --tta=1":
    
        "func=1" means the detection of sub-images and output the probability that the sub-image is fake in pixel-level.
      
        "size_idx" represents the resolution of images (the value is the index of [384, 512, 768, 1024]).
      
        "fold" indicates the model trained by different split.
      
        "tta" is the Test Time Augmentation(1-8 means the fliping and/or rotation).
      
    For "python main.py test --func=2": the "func=2" means the ensemble operation.
