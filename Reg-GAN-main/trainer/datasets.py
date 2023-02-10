import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2





class ImageDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose([transforms.ToTensor()])
        self.files_A = sorted(glob.glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\nonac\*'))
        self.files_A = self.files_A[:int(len(self.files_A)*0.80)]
        self.files_B = sorted(glob.glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\ac\*'))
        self.files_B = self.files_B[:int(len(self.files_B)*0.80)]
        self.unaligned = unaligned
        self.noise_level =noise_level
        
        
    def __getitem__(self, index):
        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # print(type(cv2.imread(self.files_A[index])))
            item_A = self.transform2(cv2.imread(self.files_A[index]).astype(np.float32))

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(cv2.imread(self.files_B[index % len(self.files_B)]).astype(np.float32))
        else:
            # if noise !=0, A and B make different transform
       
            item_A = cv2.imread(self.files_A[index])
            item_A = cv2.resize(item_A,(256,256))
            # print(item_A.shape)
            # item_A = np.transpose(item_A,(2,0,1))
            
            item_A = self.transform1(item_A)
            # print(item_A.max())
            

            item_B = cv2.imread(self.files_B[index])
            item_B = cv2.resize(item_B,(256,256))
            # print(item_B.shape)
            # item_B = np.transpose(item_B,(2,0,1))
            
            item_B = self.transform1(item_B)
            
            
            
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.unaligned = unaligned
        # self.files_A = sorted(glob.glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\nonac\*'))
        # self.files_A = self.files_A[int(len(self.files_A)*0.80):int(len(self.files_A)*0.90)]
        # self.files_B = sorted(glob.glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\ac\*'))
        # self.files_B = self.files_B[int(len(self.files_B)*0.80):int(len(self.files_B)*0.90)]

        self.files_A = sorted(glob.glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\nonac\*'))
        self.files_A = self.files_A[int(len(self.files_A)*0.90):]
        self.files_B = sorted(glob.glob(r'C:\Users\NM_RR\Desktop\SJ\data\SPECT_data\rest\ac\*'))
        self.files_B = self.files_B[int(len(self.files_B)*0.90):]
        
    def __getitem__(self, index):
        print(self.files_A[index % len(self.files_A)])
        item_A = cv2.imread(self.files_A[index % len(self.files_A)])
        name_A = os.path.basename(self.files_A[index % len(self.files_A)])
        item_A = cv2.resize(item_A,(256,256))
        
        # item_A = np.transpose(item_A,(2,0,1))
        item_A = self.transform(item_A)

        if self.unaligned:
            item_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])
            item_B = cv2.resize(item_B,(256,256))
            
            # item_B = np.transpose(item_B,(2,0,1))
            item_B = self.transform(item_B)
        else:
            print(self.files_B[index])
            item_B = cv2.imread(self.files_B[index])
            name_B= os.path.basename(self.files_B[index])
            item_B = cv2.resize(item_B,(256,256))
            
            # item_B = np.transpose(item_B,(2,0,1))
            item_B = self.transform(item_B)
        return {'A': item_A, 'B': item_B, 'name_A':name_A, 'name_B':name_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
