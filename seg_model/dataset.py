import os
import glob
import torch
import random
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from imageio import imread
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom

from args import args


class DataEnhancement(object):                                    

    def __init__(self, output_size):
        self.output_size = output_size                             

    def random_rot_flip(self, image, label):                         
        k = np.random.randint(0, 4)       

        image = np.rot90(image, k)
        label = np.rot90(label, k)

        axis = np.random.randint(0, 2)                              

        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return image, label

    def random_rotate(self, image, label):                          
        angle = np.random.randint(-20, 20)                         

        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)

        return image, label
    
    def random_fliplr(self, image, label):                         
        image = np.fliplr(image)
        label = np.fliplr(label)

        return image, label
    
    def random_flipud(self, image, label):                       
        image = np.flipud(image)
        label = np.flipud(label)

        return image, label

    def get_patch(self, img1, img2):                               

        h, w = img1.shape[:2]                                   
        stride = 96                                              

        img1p_list = []
        img2p_list = []

        for x in range(0, w - args.img_size + 1, stride):
            for y in range(0, h - args.img_size + 1, stride):

                img1_p = img1[x:x + args.img_size, y:y + args.img_size]
                img2_p = img2[x:x + args.img_size, y:y + args.img_size]

                img1p_list.append(img1_p)
                img2p_list.append(img2_p)

        return img1p_list, img2p_list

    def __call__(self, sample, crop_log, aug_log):

        image, label = sample['image'], sample['label']            

        if crop_log:                                              

            image_list, label_list = self.get_patch(image, label)   

            x, y = image_list[0].shape[-2:]
            if x != self.output_size[0] or y != self.output_size[1]:
                
                for i in range(len(image_list)):

                    image_list[i] = zoom(image_list[i].reshape(x, y), (self.output_size[0] / x, self.output_size[1] / y), order=3)
                    image_list[i] = (image_list[i] - image_list[i].min()) / (image_list[i].max() - image_list[i].min())
                    label_list[i] = zoom(label_list[i], (self.output_size[0] / x, self.output_size[1] / y), order=0)

                    if aug_log:
                        if random.random() > 0.5:                               
                            image_list[i], label_list[i] = self.random_rot_flip(image_list[i], label_list[i])
                        if random.random() > 0.5:        
                            image_list[i], label_list[i] = self.random_rotate(image_list[i], label_list[i])
                        if random.random() > 0.5:
                            image_list[i], label_list[i] = self.random_fliplr(image_list[i], label_list[i])
                        if random.random() > 0.5:
                            image_list[i], label_list[i] = self.random_flipud(image_list[i], label_list[i])

            for i in range(len(image_list)):
                image_list[i] = torch.from_numpy(image_list[i].astype(np.float32)).unsqueeze(0)
                label_list[i] = torch.from_numpy(label_list[i].astype(np.float32))

            sample = {'image': image_list, 'label': label_list}

        else:

            x, y = image.shape[-2:]
            if x != self.output_size[0] or y != self.output_size[1]:     

                image = zoom(image.reshape(x, y), (self.output_size[0] / x, self.output_size[1] / y), order=3)
                image = (image - image.min()) / (image.max() - image.min())
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

                if aug_log:
                    if random.random() > 0.5:                          
                        image, label = self.random_rot_flip(image, label)
                    if random.random() > 0.5:                                  
                        image, label = self.random_rotate(image, label)
                    if random.random() > 0.5:                                  
                        image, label = self.random_fliplr(image, label)
                    if random.random() > 0.5:                                  
                        image, label = self.random_flipud(image, label)

            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.float32))

            sample = {'image': image, 'label': label}

        return sample


class ReadData(Dataset):

    def __init__(self, data_dir, split, mask_values, crop_log=False, aug_log=False):

        self.transform = DataEnhancement(output_size=[args.img_size, args.img_size])    
        self.split = split                                                               
        self.dataset_dir = os.path.join(data_dir, split)                                 
        self.sample_img = glob.glob(os.path.join(self.dataset_dir, "images/*"))          
        self.sample_label = glob.glob(os.path.join(self.dataset_dir, "labels/*"))        
        self.sample_name = os.listdir(os.path.join(self.dataset_dir, "images/"))         
        self.mask_values = mask_values                                                   
        self.crop_log = crop_log
        self.aug_log = aug_log

    def __len__(self):

        return len(self.sample_img)
    
    def preprocess(self, mask_values, img, is_mask):                  

        h, w = img.shape[-2:]                                 

        eps = 1e-9

        if is_mask:                                             
            mask = np.zeros((h, w), dtype=np.int64)

            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:                                        # 切片为2维则升维
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))                       # 否则调整通道顺序，CHW

            img = (img - img.min() + eps) / (img.max() - img.min() + eps)

            return img

    def __getitem__(self, idx):

        img_name = self.sample_name[idx]

        if img_name.split('.')[1] == 'png':
            image = imread(self.dataset_dir + '/images/' + img_name)
            label = imread(self.dataset_dir + '/labels/' + img_name)
        elif img_name.split('.')[1] == 'dcm':
            image = sitk.ReadImage(self.dataset_dir + '/images/' + img_name)
            image = sitk.GetArrayFromImage(image).squeeze()
            label = sitk.ReadImage(self.dataset_dir + '/labels/' + img_name)
            label = sitk.GetArrayFromImage(label).squeeze()

        image = self.preprocess(self.mask_values, image, is_mask=False)
        label = self.preprocess(self.mask_values, label, is_mask=True)

        sample = {'image': image, 'label': label}

        sample = self.transform(sample, self.crop_log, self.aug_log)
        sample['case_name'] = img_name.split('.')[0]

        return sample
