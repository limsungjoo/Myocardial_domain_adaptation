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
Overall structure is that Coordconv layers and centroids are combined with the input layer of the model based on U-Net.    
Dataset was 1755 Vertebra X-ray image (Lateral view) in Severance Hospital for segmentation model.    
The proposed network was evaluated by 176 spine images and yielded an average Dice score of 0.9408.      
This network and related paper will be submitted in September, 2022.

#### Model Architecture   
![image](https://user-images.githubusercontent.com/48985628/187608509-aad9af10-031e-4bb0-a575-77b6f3144bca.png)

#### Results
<img src="https://user-images.githubusercontent.com/48985628/187634962-8abf4d0e-ad12-4824-af75-d2c513fc611b.png" width="600" height="600"/>

## Reg GAN
This network detects the centroids of the vertebrae for the localization of the vertebra using U-Net.    
Thus, the centroids which are extracted from the Centroid UNet are added to the input channel of the segmentation model(Center+Coordconv UNet).    

#### Results    
![image](https://user-images.githubusercontent.com/48985628/187630961-d99647b8-3fd3-4044-9297-a5c4675899cf.png)

## Swin-CycleGAN
Overall structure is that Coordconv layers are combined with the input layer of the model based on U-Net.     
Reference from the paper: [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247)    
Reference from the paper : [CoordConv-Unet: Investigating CoordConv for Organ Segmentation](https://doi.org/10.1016/j.irbm.2021.03.002)     

## Modified U-Net
This network is a combination of vision transforemr and U-Net for Vertebra X-ray image segmentation.     
Reference from the paper: [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## Experiments
Comparison of results for several segmentation networks    
|Networks|Dice Score|
|:--------:|:----:|
|Diffusion models|0.9514|
|Reg GAN|0.9494|
|Swin-CycleGAN|0.9152|
|Modified U-Net|0.9243|



