Myocardial Domain Adaptation
=====================
I have studied various methods to solve domain adaptation for Myocardial 3D images.
The following models have been developed and the results of the models were compared.   
Submitted for a paper titled "Clinical Feasibility of Deep Learning-based Attenuation Correction Models for 201Tl-Myocardial Perfusion Single-Photon Emission Computed Tomography"
![ajou](https://github.com/limsungjoo/Myocardial_domain_adaptation/assets/48985628/2277866b-a39b-4656-8edb-c6d69c0f291c)

* [Requirements](#requirements)
* [Reg GAN](#reg-gan)
* [Diffusion models](#diffusion-models)
* [Swin-CycleGAN](#swin-cyclegan)
* [Modified U-Net](#modified-u-net)
* [Experiments](#experiments)
----------------------

## Requirements
* pytorch: 1.10.2+cu113
* torchvision: 0.11.3+cu113
* CUDA version: 12.0
* Python: 3.8.8
## Reg GAN

#### Results    
![color](https://user-images.githubusercontent.com/48985628/222372702-295cdecc-a9c8-4cd0-8714-7d68d95e45b1.png)



## Diffusion models

#### Results
![diffusion](https://user-images.githubusercontent.com/48985628/222371309-571529b0-f74b-4c22-a111-83a45d1ad9cd.png)


## Swin-CycleGAN

#### Results
![gray_swin](https://user-images.githubusercontent.com/48985628/222370782-be7dca3d-2084-48b3-9cd7-928e1b13fb98.png)


## Modified U-Net
#### Results
![color_unet](https://user-images.githubusercontent.com/48985628/222372728-d6be25b7-8af9-4128-aaf6-358ce71d63c5.png)

#### Results
![gray_unet](https://user-images.githubusercontent.com/48985628/222370282-0ffc2db7-b7ed-45ad-bb93-5b1ebb163981.png)


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



