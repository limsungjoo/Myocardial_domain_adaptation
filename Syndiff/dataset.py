import torch.utils.data
import numpy as np
import random
import cv2


def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2'):

    target_file = input_path + "/data_{}_{}.npy".format(phase, contrast1)
    data_fs_s1=LoadDataSet(target_file)
    
    target_file = input_path + "/data_{}_{}.npy".format(phase, contrast2)
    data_fs_s2=LoadDataSet(target_file)

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))  
    return dataset 



#Dataset loading from load_dir and converintg to 256x256 
def LoadDataSet(load_dir,  padding = True, Norm = True):
    print(load_dir)
    f = np.load(load_dir)
    print(f.shape)
    
    # f = h5py.File(load_dir,'r') 
    # if f.ndim==3:
    #     data=np.expand_dims(np.transpose(f,(0,2,1)),axis=1)
    # else:
    #     data=np.transpose(f,(1,0,3,2))
    data=f.astype(np.float32) 
    # if padding:
    #     pad_x=int((256-data.shape[2])/2)
    #     pad_y=int((256-data.shape[3])/2)
    #     print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
    #     data=np.pad(data,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))   
    if Norm:    
        data=(data-0.5)/0.5      
    return data
