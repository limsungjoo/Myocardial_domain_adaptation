import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from glob import glob
import SimpleITK as sitk

from utils.preprocessing import centercropping
from utils.transforms import image_windowing, image_minmax, mask_binarization, augment_imgs_and_masks, center_crop
import matplotlib.pyplot as plt

class VertebralDataset(Dataset):
    def __init__(self, opt, is_Train=True, augmentation=True):
        super(VertebralDataset, self).__init__()

        self.augmentation = augmentation
        self.opt = opt
        self.is_Train = is_Train
        
        if is_Train:

            # self.files_A = sorted(glob(r"C:\Users\NM_RR\Desktop\SJ\Reg-GAN-main\data\train2D\A\*"))
            # self.files_B = sorted(glob(r"C:\Users\NM_RR\Desktop\SJ\Reg-GAN-main\data\train2D\B\*"))

            self.files_A = sorted(glob('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(nonac)/sh/*'))
            self.files_A = self.files_A[:int(len(self.files_A)*0.80)]

            self.files_B = sorted(glob('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(ac)/sh/*'))
            self.files_B = self.files_B[:int(len(self.files_B)*0.80)]
            # self.mask_list = sorted(glob(opt.data_root))

            # print(self.mask_list)
            # self.files_A = self.files_A[:int(len(self.files_A)*0.80)]
            # self.files_B = self.files_B[:int(len(self.files_B)*0.80)]
            # print(self.mask_list)
            print(len(self.files_B),"dataset: Training")
        
        else:
            # self.files_A = sorted(glob(r"C:\Users\NM_RR\Desktop\SJ\Reg-GAN-main\data\val2D\A\*"))
            # self.files_B = sorted(glob(r"C:\Users\NM_RR\Desktop\SJ\Reg-GAN-main\data\val2D\B\*"))

            # test

            # self.files_A = sorted(glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\nonac\*'))
            # self.files_A = self.files_A[int(len(self.files_A)*0.80):int(len(self.files_A)*0.90)]

            # self.files_B = sorted(glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\ac\*'))
            # self.files_B = self.files_B[int(len(self.files_B)*0.80):int(len(self.files_B)*0.90)]
            
            self.files_A = sorted(glob('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(nonac)/sh/*'))
            self.files_A = self.files_A[int(len(self.files_A)*0.80):int(len(self.files_A)*0.90)]
            # self.files_A = self.files_A[int(len(self.files_A)*0.90):]

            self.files_B = sorted(glob('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(ac)/sh/*'))
            self.files_B = self.files_B[int(len(self.files_B)*0.80):int(len(self.files_B)*0.90)]
            # self.files_B = self.files_B[int(len(self.files_B)*0.90):]
            print(len(self.files_B),"dataset: Vailidation")
                
        
        self.len = len(self.files_A)

        # self.augmentation = augmentation
        # self.opt = opt

        # self.is_Train = is_Train


    def __getitem__(self, index):
        # Load Image and Mask
        mask_path = self.files_B[index]
        
        
        # xray_path = mask_path.replace('/Label/', '/Dataset/').replace('_label.png', '.png')

        xray_path = self.files_A[index]

        img_name = os.path.basename(xray_path)
        mask_name = os.path.basename(mask_path)
        # print(mask_path)
        # print(xray_path)
        # print('img:',xray_path)
        # print('msk:',mask_path)
        mask = cv2.imread(mask_path,0)
        img = cv2.imread(xray_path,0)
        img =cv2.resize(img,(256,256))
        mask =cv2.resize(mask,(256,256))
        img = img[np.newaxis,:,:]
        mask = mask[np.newaxis,:,:]
        # img = np.transpose(img)
        # mask = np.transpose(mask)
        
        # img = cv2.equalizeHist(img)
        # img,mask = centercropping(img,mask)
        

        # HU Windowing
        # img = image_windowing(img, self.opt.w_min, self.opt.w_max)

        
        
        
        # MINMAX to [0, 1]
        img = img / 255.
        mask = mask/255.

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        # Mask Binarization (0 or 1)
        # mask = mask_binarization(mask)

        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/img/'+str(index)+'.jpg',img)
        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/msk/'+str(index)+'.png',mask)
        # Add channel axis
        # img = img[None, ...].astype(np.float32)
        # mask = mask[None, ...].astype(np.float32)
                
        # Augmentation
        if self.augmentation:
            img, mask = augment_imgs_and_masks(img, mask, self.opt.rot_factor, self.opt.scale_factor, self.opt.trans_factor, self.opt.flip)

        return img, mask, img_name, mask_name
        
    def __len__(self):
        return self.len


