import argparse
import sys
import os
from PIL import Image
import cv2
from PIL import Image
import SimpleITK as sitk
import torchvision.transforms as transforms
import torchvision.utils as v_utils
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch
from Unetencoder import Generator
from uvcgan.config      import Args
from uvcgan.cgan        import construct_model
from uvcgan.config import Config
from uvcgan.models.generator import ViTUNetGenerator
# from CycleGAN import Generator
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Definition of variables ######
# Networks
netG_A2B = ViTUNetGenerator(384, 6,6,1536,384,'gelu','layer',(1,128,128),[48, 96, 192, 384],'leakyrelu','instance').to(device)
netG_B2A = ViTUNetGenerator(384, 6,6,1536,384,'gelu','layer',(1,128,128),[48, 96, 192, 384],'leakyrelu','instance').to(device)

# netG_A2B = torch.nn.DataParallel(netG_A2B)
# netG_B2A=torch.nn.DataParallel(netG_B2A)

# Load state dicts
netG_A2B.load_state_dict(torch.load(r'C:\Users\NM_RR\Desktop\SJ\Cycle-GAN\output\rest\netG_A2B\370_loss_epoch_netG_A2B.pkl', map_location="cuda:0"))
netG_B2A.load_state_dict(torch.load(r'C:\Users\NM_RR\Desktop\SJ\Cycle-GAN\output\rest\netG_B2A\370_loss_epoch_netG_B2A.pkl', map_location="cuda:0"))
print(netG_A2B)
# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(1, 1, 128,128) # (batchsize, output channel, size, size)
input_B = Tensor(1, 1, 128,128)

# Dataset loader

image = sorted(os.listdir('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(nonac)/sh/'))

image = image[int((len(image))*0.8):]
label = sorted(os.listdir('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(ac)/sh/'))

label = label[int((len(label))*0.8):]
print(len(image),len(label))


# for i in range(0,10000):
#     random.shuffle(label)
#     if 'busan' == label[0]:
#         break
#     else :
#         continue
# label = label[:58]
# print(len(label))
# print(label)
class CycleGanData_test(Dataset):
    def __init__(self,testA_path,testB_path,transform):
        self.testA_path = testA_path
        self.testB_path = testB_path
        self.transform = transform
        
    def __len__(self):
        return len(image)
    
    def testA(self,testA_path):
        testA = cv2.imread('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(nonac)/sh/'+testA_path,0)
        testA = cv2.resize(testA,(128,128))
        # testA = sitk.ReadImage('/home/vfuser/sungjoo/data/Lateral_deepnoid/spine/'+testA_path)
        # testA = testA.resize((296,420))
        # testA = self.transform(testA)
        # testA = sitk.GetArrayFromImage(testA)
        # testA = cv2.cvtColor(testA, cv2.COLOR_BGR2GRAY)
        # trainA = image_minmax(trainA)
        # IMG_SIZE = 256

        # ori_size = trainA.shape

        # h,w = trainA.shape
        
        # bg_img = np.zeros((1024,512))

        # if w>h:
        #     x=512
        #     y=int(h/w *x)
        # else:
        #     y=1024
        #     x=int(w/h *y)

        #     if x >512:
        #         x =512
        #         y= int(h/w *x)
        
        # img_resize = cv2.resize(trainA, (x,y))

        # xs = int((512 - x)/2)
        # ys = int((1024-y)/2)
        # bg_img[ys:ys+y,xs:xs+x]=img_resize

        
        # cv2.imwrite('/home/vfuser/sungjoo/Cycle-GAN/output/testA/'+testA_path,testA)
        testA = testA / 255.
        
        testA = self.transform(testA)
        return testA
    
    def testB(self,testB_path):
        testB = cv2.imread('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(ac)/sh/'+testB_path,0)
        testB = cv2.resize(testB,(128,128))
       
        testB = testB / 255.
        testB = self.transform(testB)
        return testB
    
    def __getitem__(self,index):
        testA = self.testA(self.testA_path[index])
        testB = self.testB(self.testB_path[index])
        testA_name = self.testA_path[index]  
        testB_name = self.testB_path[index]   
        return {'testA': testA,
                'testB': testB,
                'testA_name' : testA_name,
                'testB_name' : testB_name}

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((256,256), Image.BICUBIC),
    # transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# DataSet, DataLoader
Dataset = CycleGanData_test(testA_path=image,testB_path=label,transform=transform)
# print(Dataset[2]['trainA'])

dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1,
                                          shuffle=False, num_workers=0,drop_last=True)

###### Testing######

# Create output dirs if they don't exist
# if not os.path.exists('output/A_image'):
#     os.makedirs('output/A_image')
# if not os.path.exists('output/B_image'):
#     os.makedirs('output/B_image')

# for i, batch in enumerate(dataloader):
#     # Set model input
#     real_A = Variable(input_A.copy_(batch['testA']))
#     real_B = Variable(input_B.copy_(batch['testB']))

#     # Generate output
#     fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
#     fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

#     # Save image files
#     batch_tensorA = torch.cat((real_A, fake_B), dim=2)
#     batch_tensorB = torch.cat((real_B, fake_A), dim=2)
    
#     grid_imgA = v_utils.make_grid(batch_tensorA) # padding = 1, nrow = 4
#     grid_imgB = v_utils.make_grid(batch_tensorB)

#     v_utils.save_image(grid_imgA, '/home/vfuser/sungjoo/Cycle-GAN/output/A_image/%04d.jpg' % (i+1))
#     v_utils.save_image(grid_imgB, '/home/vfuser/sungjoo/Cycle-GAN/output/B_image/%04d.jpg' % (i+1))

#     sys.stdout.write('\rGenerated images %04d of %04d'%(i+1, len(dataloader)))
# sys.stdout.write('\n')
#PSNR#
import numpy 
import numpy as np
import math
from IQA_pytorch import SSIM
def psnr(img1, img2):

    mse = numpy.mean( (img1 - img2) ** 2 ) #MSE 구하는 코드

    print("mse : ",mse)

    if mse == 0:

        return 100

    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def MAE_f(fake,real):
        # x,y = np.where(real!= -1)  # coordinate of target points
        #points = len(x)  #num of target points
        mae = np.abs( (real - fake)).mean()
        # mae = np.abs(fake[x,y]-real[x,y]).mean()
            
        return mae/2

def PSNR_f(fake,real):
    #    x,y = np.where(real!= -1)
    #    mse = np.mean(((fake[x][y]+1)/2. - (real[x][y]+1)/2.) ** 2 )
       mse = np.mean( (real - fake) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse)) 
k=0
total_mae = 0
total_psnr = 0
SSIM_1 = 0
num = 0
total = 0
from torchsummary import summary
for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['testA']))
    real_B = Variable(input_B.copy_(batch['testB']))
    name_A = batch['testA_name']
    name_B = batch['testB_name']
    # print(name_A)
    # Generate output
    fake_B = netG_A2B(real_A).data
    # fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    # batch_tensorA = torch.cat((real_A, fake_B), dim=2)
    # batch_tensorB = torch.cat((real_B, fake_A), dim=2)
    batch_tensorA = real_A
    batch_tensorB = fake_B
    batch_tensorC = real_B


    # grid_imgA = v_utils.make_grid(batch_tensorA) # padding = 1, nrow = 4
    # grid_imgB = v_utils.make_grid(batch_tensorB)
    # grid_imgC = v_utils.make_grid(real_B)

    # v_utils.save_image(grid_imgA, 'C:/Users/NM_RR/Desktop/SJ/Cycle-GAN/output/rest/visualized/realA/'+ name_A[0])
    # v_utils.save_image(grid_imgB, 'C:/Users/NM_RR/Desktop/SJ/Cycle-GAN/output/rest/visualized/fakeB/'+name_A[0])
    # v_utils.save_image(grid_imgC, 'C:/Users/NM_RR/Desktop/SJ/Cycle-GAN/output/rest/visualized/realB/'+name_B[0])
    # v_utils.save_image(grid_imgA, '/home/vfuser/sungjoo/Cycle-GAN/output/A_image/%04d.jpg' % (i+1))
    # v_utils.save_image(grid_imgA, '/home/vfuser/sungjoo/Cycle-GAN/output/deepnoid_A/'+ name_A[0])
    # v_utils.save_image(grid_imgB, '/home/vfuser/sungjoo/Cycle-GAN/output/deepnoid_B/'+name_A[0])
    # v_utils.save_image(grid_imgC, '/home/vfuser/sungjoo/Cycle-GAN/output/GE_Real/'+name_B[0])
    
    sys.stdout.write('\rGenerated images %04d of %04d'%(i+1, len(dataloader)))
    real = real_B.cpu().numpy().squeeze()
    fake = fake_B.cpu().numpy().squeeze()
    real_A = real_A.cpu().numpy().squeeze()

    model_ssim = SSIM(channels=1)
    ssim = model_ssim(fake_B,real_B, as_loss=False)

    mae = MAE_f(fake,real)
    psnr = PSNR_f(fake,real)
    total_mae += mae
    total_psnr += psnr
    SSIM_1 += ssim
    print(psnr)

    print(real_A.shape)
    # img_n = np.transpose(real_A,(1,2,0))
    # pred_n = np.transpose(fake_B,(1,2,0))
    # mask_n = np.transpose(real_B,(1,2,0))

    cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/Cycle-GAN/output/rest/visualized/realA/'+name_A[0],real_A*255)
    cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/Cycle-GAN/output/rest/visualized/fakeB/'+name_A[0],fake*255)
    cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/Cycle-GAN/output/rest/visualized/realB/'+name_B[0],real*255)
    # psnr_s = psnr(real,fake)
    # print(psnr_s)
    # total+=round(psnr_s,2)
    num += 1
    # k+=1
sys.stdout.write('\n')
print(k)
print ('MAE:',total_mae/num)
print ('PSNR:',total_psnr/num)
print ('SSIM:',SSIM_1/num)
# avg = total/k
# print(avg)
###################################

