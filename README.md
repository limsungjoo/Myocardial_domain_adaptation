Myocardial Domain Adaptation
=====================
I have studied various methods to solve domain adaptation for Myocardial 3D images.
The following models have been developed and the results of the models were compared.      
* [Requirements](#requirements)
* [Diffusion models](#diffusion-models)
* [Reg GAN](#reg-gan)
* [Swin-CycleGAN](#swin-cyclegan)
* [Modified U-Net](#modified-u-net)
* [Experiments](#experiments)
----------------------

## Requirements
* pytorch: 1.10.2+cu113
* torchvision: 0.11.3+cu113
* CUDA version: 12.0
* Python: 3.8.8

## Diffusion models

#### Model Architecture   
![image]()

#### Results
<img src="" width="600" height="600"/>

## Reg GAN
  

#### Results    
![image]()

## Swin-CycleGAN
 

## Modified U-Net


## Experiments
Comparison of results for several domain adaptation networks (Gray scale images)
|Networks|PSNR|SSIM|
|:--------:|:----:|:----:|
|Diffusion models|28.3914|0.9706|
|Swin-CycleGAN|28.5760|0.9602|
|Modified U-Net|33.6580|0.9903|

Comparison of results for several domain adaptation networks (Color scale images)
|Networks|PSNR|SSIM|
|:--------:|:----:|:----:|
|Reg GAN|18.7497|0.7910|
|Modified U-Net|20.1862|0.8378|



