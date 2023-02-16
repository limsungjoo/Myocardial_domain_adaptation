import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import time
import datetime
from visdom import Visdom
import torch
from torch.autograd import Variable
import sys
from utils import AverageMeter
from utils.metrics import DiceCoef
from utils.losses import iou_modified, avg_precision
from utils.losses import *
from utils.psave import *
import cv2
from matplotlib import pyplot as plt 
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from IQA_pytorch import SSIM

import torch
from torch.autograd import Variable

from utils import AverageMeter
from utils.metrics import DiceCoef
from utils.losses import iou_modified, avg_precision
from utils.psave import *
# import pytorch_ssim


from matplotlib import pyplot as plt

import numpy
from scipy.ndimage.morphology import generate_binary_structure,binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import label, find_objects

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data
            else:
                self.losses[loss_name] += losses[loss_name].data

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=torch.cuda.FloatTensor([self.epoch]), Y=torch.cuda.FloatTensor([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=torch.cuda.FloatTensor([self.epoch]), Y=torch.cuda.FloatTensor([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        


def train(net, dataset_trn, optimizer, criterion, epoch, opt,train_writer):
    
    print("Start Training...")
    net.train()

    losses, total_dices, total_iou = AverageMeter(), AverageMeter(), AverageMeter()
    # logger = Logger(epoch, len(dataset_trn))
    for it, (img, mask,img_name,mask_name) in enumerate(dataset_trn):
        
        # Optimizer
        optimizer.zero_grad()
        # print(img_name,mask_name)
        # Load Data
        img, mask = torch.Tensor(img), torch.Tensor(mask)
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        mse_loss = torch.nn.MSELoss()
        mse_loss = mse_loss(pred, mask)

        # ssim_value = pytorch_ssim.ssim(mask, pred).data[0]
        # s_pred = Variable( pred,  requires_grad=True)
        # s_mask = Variable( mask, requires_grad = False)
        # ssim_value = pytorch_ssim.ssim(s_mask, s_pred).data[0]
        ssim_loss = SSIM(channels=3)

        ssim_out = ssim_loss(pred, mask, as_loss=True)
        # cos = torch.nn.CosineEmbeddingLoss
        # cos_loss = cos(pred,mask)
        
        loss = mse_loss+ssim_out

        # pred = pred.sigmoid()
        # Backward and step
        loss.backward()
        
        
            
            
            
        optimizer.step()

        # Progress report (http://localhost:8097) (python -m visdom.server)
        # logger.log({'loss':loss}, 
        #            images={'image': img, 'pred': pred, 'GT': mask})
        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred, mask)
        total_dices.update(dice.item(), img.size(0))
        
        # # Convert to Binary
        # zeros = torch.zeros(pred.size())
        # ones = torch.ones(pred.size())
        # pred = pred.cpu()

        # pred = torch.where(pred > 0.9, ones, zeros).cuda() # threshold 0.99

        # # Calculation IoU Score
        # iou_score = iou_modified(pred, mask,opt)

        # total_iou.update(iou_score.mean().item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f \n'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg))

    print(">>> Epoch[%3d/%3d] | Training Loss : %.4f \n"
        % (epoch+1, opt.max_epoch, losses.avg))

    train_writer.add_scalar("train/loss", losses.avg, epoch+1)
    # train_writer.add_scalar("train/dice", total_dices.avg, epoch+1)
    # train_writer.add_scalar("train/IoU", total_iou.avg, epoch+1)


def validate(dataset_val, net, criterion, epoch, opt, best_iou, best_epoch,train_writer):
    print("Start Evaluation...")
    net.eval()
    
    # Result containers
    losses, total_dices, total_iou = AverageMeter(), AverageMeter(), AverageMeter()

    for it, (img, mask,img_name,mask_name) in enumerate(dataset_val):
        # Load Data
        print(img_name,mask_name)
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        mse_loss = torch.nn.MSELoss()
        mse_loss = mse_loss(pred, mask)

       

        ssim_loss = SSIM(channels=3)

        ssim_out = ssim_loss(pred, mask, as_loss=True)
        # cos = torch.nn.CosineEmbeddingLoss
        # cos_loss = cos(pred,mask)
        # ssim_loss= -ssim_loss(s_pred, s_mask)
        # cos = torch.nn.CosineEmbeddingLoss()
        # cos_loss = cos(pred,mask)

        loss = mse_loss+ssim_out
        # loss = criterion(pred, mask)

        # pred = pred.sigmoid()

        # # Calculation Dice Coef Score
        # dice = DiceCoef(return_score_per_channel=False)(pred, mask)
        # total_dices.update(dice.item(), img.size(0))
        
        # # Convert to Binary
        # zeros = torch.zeros(pred.size())
        # ones = torch.ones(pred.size())
        # pred = pred.cpu()

        # pred = torch.where(pred > 0.9, ones, zeros).cuda()
        
        # # Calculation IoU Score
        # iou_score = iou_modified(pred, mask,opt)

        # total_iou.update(iou_score.mean().item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        # if (it==0) or (it+1) % 10 == 0:
        #     print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice %.4f | Iou %.4f'
        #         % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, total_dices.avg, total_iou.avg))

    print(">>> Epoch[%3d/%3d] | Test Loss : %.4f "
        % (epoch+1, opt.max_epoch, losses.avg))

    train_writer.add_scalar("valid/loss", losses.avg, epoch+1)
    

    # Update Result
    

        # # Remove previous weights pth files
        # for path in glob('%s/*.pth' % opt.exp):
        #     os.remove(path)

    model_filename = '%s/epoch_%04d_loss_%.8f.pth' % (opt.exp, epoch+1,  losses.avg)

    # Single GPU
    if opt.ngpu == 1:
        torch.save(net.state_dict(), model_filename)
    # Multi GPU
    else:
        torch.save(net.module.state_dict(), model_filename)

    # print('>>> Current best: IoU: %.8f in %3d epoch\n' % (best_iou, best_epoch+1))
    
    return 0, best_epoch



def evaluate(dataset_val, net, opt):
    
    print("Start Evaluation...")
    net.eval()

    iou_scores = []
    i=0
    k=0
    total = 0
    total_mae = 0
    total_psnr = 0
    SSIM_1 = 0
    num = 0
    for idx, (img, mask,img_name,mask_name) in tqdm(enumerate(dataset_val)):
        # Load Data
        img = torch.Tensor(img).float()
        mask = torch.Tensor(mask).float()
        if opt.use_gpu:
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
        # Predict
        with torch.no_grad():
            
            pred = net(img)
            
            
            model_ssim = SSIM(channels=3)
            ssim = model_ssim(pred,mask, as_loss=False)

            img_n = img.detach().cpu().numpy().squeeze()
            pred_n = pred.detach().cpu().numpy().squeeze()
            mask_n = mask.detach().cpu().numpy().squeeze()
            
            
            mae = MAE_f(pred_n,mask_n)
            psnr = PSNR_f(pred_n,mask_n)
            total_mae += mae
            total_psnr += psnr
            SSIM_1 += ssim
            print(psnr)
            # print(img_n.shape)
            # img_n = img_n[:,:,np.newaxis]
            # pred_n = pred_n[:,:,np.newaxis]
            # mask_n = mask_n[:,:,np.newaxis]
            # img_n = np.transpose(img_n,(1,2,0))
            # pred_n = np.transpose(pred_n,(1,2,0))
            # mask_n = np.transpose(mask_n,(1,2,0))
            # plt.imshow(fake_B)
            # plt.savefig('C:/Users/NM_RR/Desktop/SJ/Reg-GAN-main/output/visualized/rest_fake_B/'+str(num)+'.jpg')
            cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/U-Net/output/rest_dicom/sh/visualized/realA/'+img_name[0],img_n*255)
            cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/U-Net/output/rest_dicom/sh/visualized/fake/'+mask_name[0],pred_n*255)
            cv2.imwrite('C:/Users/NM_RR/Desktop/SJ/U-Net/output/rest_dicom/sh/visualized/realB/'+mask_name[0],mask_n*255)
            num += 1

    print ('MAE:',total_mae/num)
    print ('PSNR:',total_psnr/num)
    print ('SSIM:',SSIM_1/num)
            # if iou_score < 0.75:
            ###### Plot & Save Figure #########
            

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