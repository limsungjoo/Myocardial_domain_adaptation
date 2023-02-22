import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from IQA_pytorch import SSIM
import cv2

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def inception_score(test_dataset, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    # test
    # N = len(imgs)

    # assert batch_size > 0
    # assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        y = inception_model(x)
        return F.softmax(y).data.cpu().numpy()

    # Get predictions
    # preds = np.zeros((N, 1000))
    
    iou_scores = []
    i=0
    k=0
    total = 0
    total_mae = 0
    total_psnr = 0
    SSIM_1 = 0
    num = 0
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds = get_pred(batchv)
        # preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
        img = torch.Tensor(batchv[0]).float()
        mask = torch.Tensor(batchv[1]).float()

        # Predict
        with torch.no_grad():
            preds = get_pred(img)
            
            
            
            model_ssim = SSIM(channels=3)
            ssim = model_ssim(preds,mask, as_loss=False)

            img_n = img.detach().cpu().numpy().squeeze()
            pred_n = preds.detach().cpu().numpy().squeeze()
            mask_n = mask.detach().cpu().numpy().squeeze()
            
            
            mae = MAE_f(pred_n,mask_n)
            psnr = PSNR_f(pred_n,mask_n)
            total_mae += mae
            total_psnr += psnr
            SSIM_1 += ssim
            print(psnr)
            cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/Palette-Image-to-Image-Diffusion-Models-main/experiments/visualized/realA/'+str(num)+'.jpg',img_n*255)
            cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/Palette-Image-to-Image-Diffusion-Models-main/experiments/visualized/fake/'+str(num)+'.jpg',pred_n*255)
            cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/Palette-Image-to-Image-Diffusion-Models-main/experiments/visualized/realB/'+str(num)+'.jpg',mask_n*255)
            num+=1
    # Now compute the mean kl-div
    # split_scores = []

    # for k in range(splits):
    #     part = preds[k * (N // splits): (k+1) * (N // splits), :]
    #     py = np.mean(part, axis=0)
    #     scores = []
    #     for i in range(part.shape[0]):
    #         pyx = part[i, :]
    #         scores.append(entropy(pyx, py))
    #     split_scores.append(np.exp(np.mean(scores)))

    return total_psnr/num, SSIM_1/num,total_mae/num


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