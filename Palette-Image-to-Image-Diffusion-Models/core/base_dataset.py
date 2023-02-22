import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('L')

from glob import glob 

class BaseDataset(data.Dataset):
    def __init__(self, data_root, image_size=[256, 256], loader=pil_loader):
        imgs = sorted(glob('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(nonac)/sh/*'))
        masks = sorted(glob('C:/Users/NM_RR/Desktop/SJ/data/MPS_recon_data/REST(ac)/sh/*'))
        self.imgs = imgs[int(len(imgs)*0.90):]
        self.masks = masks[int(len(masks)*0.90):]
        # self.imgs = make_dataset(data_root)
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        mask_path = self.masks[index] 
        img = self.tfs(self.loader(path))
        mask = self.tfs(self.loader(mask_path))

        # path = self.imgs[index]
        # img = self.tfs(self.loader(path))
        return img,mask

    def __len__(self):
        return len(self.imgs)
