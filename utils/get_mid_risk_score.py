import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from args import args
from dataset import ReadDataset
from utils.seg_patient_image import seg_patient_image
from utils.split_tumor import split_tumor, find_max_tumor

                                  
def get_mid_risk_score(net, segnet, device, data_path):

    tr_dataset = ReadDataset(data_path[1] + 'train data/', data_path[0], rot_log=False)
    dataloader = DataLoader(tr_dataset, shuffle=False, drop_last=True, batch_size=1, num_workers=4, pin_memory=True)

    risk_list = []

    if not os.path.exists('./outputs/train_risk_score_table.csv'):

        title = ['Index', 'Patient', 'Survival Risk Score']
        risk_df = pd.DataFrame(index=[], columns=title)

        net.eval()  

        with torch.inference_mode():
            with tqdm(total=len(dataloader), desc='Mid risk', unit='iteration', ncols=100) as pbar:
                for idx, batch in enumerate(dataloader):

                    image, name = batch['image'], batch['name']
                    image = image.to(device=device, dtype=torch.float32)
                    seg_mask = seg_patient_image(segnet, image)
                    max_image, max_mask = find_max_tumor(image, seg_mask, batch['name'])   
                    tumorimg, _ = split_tumor(max_image, max_mask)

                    outputs = net(tumorimg)

                    risk_list.append(outputs.detach().cpu().numpy()[0][0])     

                    tmp = pd.Series([idx+1, name[0], outputs.detach().cpu().numpy()[0][0]], index=title)
                    risk_df = risk_df._append(tmp, ignore_index=True)                    
                    risk_df.to_csv('./outputs/train_risk_score_table.csv', index=False)  
                    
                    pbar.update(1)       

        risk_array = np.array(risk_list)
        mid_risk_score = np.median(risk_array)                              

    else:

        risk_df = pd.read_csv('./outputs/train_risk_score_table.csv')
        risk_score_array = np.array(risk_df['Survival Risk Score'])
        mid_risk_score = np.median(risk_score_array)       
                                                               
    return mid_risk_score




