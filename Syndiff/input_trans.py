import numpy as np
import os
from glob import glob
import cv2

source_list = glob('C:/Users/user/Desktop/SJ/data/low/*')[:4]
target_list = glob('C:/Users/user/Desktop/SJ/data/high/*')[:4]
val_source_list = glob('C:/Users/user/Desktop/SJ/data/low/*')[4:]
print(val_source_list)
val_target_list = glob('C:/Users/user/Desktop/SJ/data/high/*')[4:]

save_source=np.array([])
save_target=np.array([])
for i in source_list:
    print(i)
    source = cv2.imread(i)
    
    save_source=np.append(save_source,source)
np.save('C:/Users/user/Desktop/SJ/data/input_data/data_train_contrast1.npy',save_source)

for j in target_list:
    target = cv2.imread(j)
    # target = np.load(j,allow_pickle=True)
    save_target=np.append(save_target,target)
np.save('C:/Users/user/Desktop/SJ/data/input_data/data_train_contrast2.npy',save_target)

val_source = cv2.imread(val_source_list[0])
val_target = cv2.imread(val_target_list[0])
np.save('C:/Users/user/Desktop/SJ/data/input_data/data_val_contrast1.npy',val_source)
np.save('C:/Users/user/Desktop/SJ/data/input_data/data_val_contrast2.npy',val_target)