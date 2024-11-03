import os
import cv2
import torch
import random
import imageio
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from args import args


class DataEnhancement(object):                              

    def __init__(self, output_size, rot_log=False):
        self.output_size = output_size                       
        self.rot_log = rot_log                               

    def rotate_image(self, image):                           
        if random.random() < 0.5:
            k = np.random.randint(0, 4)                           
            image = np.rot90(image, k)
        else:
            angle = np.random.randint(-30, 30)                    
            image = ndimage.rotate(image, angle, order=0, reshape=False)
        
        return image
    
    def random_flip(self, image):                            
        if random.random() < 0.5:
            image = np.fliplr(image)
        else:                                  
            image = np.flipud(image)

        return image

    def random_translation(self, image):                      
        height, width, _ = image.shape
        shift_x = int(width * np.random.uniform(-0.1, 0.1))
        shift_y = int(height * np.random.uniform(-0.1, 0.1))

        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        translated_image = cv2.warpAffine(image, matrix, (width, height))

        return translated_image

    def __call__(self, image_array):

        h, w = image_array.shape[-2:]                         

        if image_array.ndim == 2:
            image_array = image_array[np.newaxis, :, :]

        if h != self.output_size[0] or w != self.output_size[1]:
            new_image_list = []

            for i in range(image_array.shape[0]):
                new_image_list.append(zoom(image_array[i], (self.output_size[0] / h, self.output_size[1] / w), order=3))

        new_image_array = np.array(new_image_list)

        if self.rot_log == True:
            new_image_array = new_image_array.transpose((1, 2, 0)) 

            operations = [self.rotate_image, self.random_flip, self.random_translation]
            np.random.shuffle(operations)
            for op in operations[:np.random.randint(1, len(operations) + 1)]:
                new_image_array = op(new_image_array)

            new_image_array = new_image_array.transpose((2 ,0, 1))

        new_image_tensor = torch.from_numpy(new_image_array.astype(np.float32))

        new_image_tensor = (new_image_tensor - new_image_tensor.min()) / (new_image_tensor.max() - new_image_tensor.min())

        return new_image_tensor


class ReadDataset(Dataset):

    def __init__(self, patient_path, clinic_path, rot_log=False):

        super(ReadDataset, self).__init__()

        self.patient_path = patient_path                                        
        self.clinic_path = clinic_path                                          
        self.patient_name_list = os.listdir(self.patient_path)                   
        self.rot_log = rot_log                                                   
        self.data_transform = DataEnhancement(output_size=(args.img_size, args.img_size), rot_log=self.rot_log)          

    def __len__(self):
        return len(self.patient_name_list)                                      

    def __getitem__(self, idx):

        image_mask_path = self.patient_path + self.patient_name_list[idx]         
        image_name_list = os.listdir(image_mask_path + '/image/')                

        if len(image_name_list) != 18:
            if len(image_name_list) > 18:
                leave_num = len(image_name_list) - 18
                subtrahend = int(leave_num / 2)
                remainder = leave_num % 2
                image_name_list = image_name_list[subtrahend:len(image_name_list) - (subtrahend + remainder)]
            else:
                last_value = image_name_list[-1] 
                image_name_list = image_name_list + [last_value] * (18 - len(image_name_list))

        image_list = []
        if os.listdir(image_mask_path + '/image/')[0].endswith('.png'):
            for image_name in image_name_list:
                image = imageio.imread(image_mask_path + '/image/' + image_name)
                image_list.append(image)
                image_array = np.array(image_list)
        else:
            for image_name in image_name_list:
                image = sitk.GetArrayFromImage(sitk.ReadImage(image_mask_path + '/image/' + image_name)).squeeze()
                image_list.append(image)
                image_array = np.array(image_list)

        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())   

        if self.data_transform:
            image_tensor = self.data_transform(image_array)                                         

        clinic_data = pd.read_excel(self.clinic_path + 'clinical data table...')
        clinic_data['patient id'] = clinic_data['patient id'].astype(str).str.zfill(8)

        clinic_data = clinic_data[clinic_data['patient id'] == self.patient_name_list[idx].split('_')[0]]

        selected_columns = ['PFS', 'PFS_e', 'OS', 'OS_e']
        clinic_data = clinic_data[selected_columns]

        clinic_array = np.array(clinic_data.values)

        clinic_tensor = torch.from_numpy(clinic_array.astype(float)).squeeze()

        data = {'name': self.patient_name_list[idx], 'image': image_tensor, 'clinic': clinic_tensor}

        return data



