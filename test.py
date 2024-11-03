import os
import sys
import time
import torch
import random
import logging

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from args import args
from dataset import ReadDataset
from networks.network import predict_net
from seg_model.networks.unet_model import UNet
from utils.seg_patient_image import seg_patient_image
from utils.get_mid_risk_score import get_mid_risk_score
from utils.split_tumor import split_tumor, find_max_tumor
from utils.c_index import compute_concordance_index, compute_concordance_index_bootstrap_ci


def predict(model, segmodel, device, data_path, outputs_path, threshold):

    test_dataset = ReadDataset(data_path[1] + 'test data/', data_path[0], rot_log=False)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, batch_size=1, num_workers=4, pin_memory=True)

    logging.info(f'''Starting testing:
                Batch size:      {1}
             learning rate:      {args.learning_rate}
                 Test size:      {len(test_loader) * 1}
                    Device:      {device.type}
                 Threshold:      {threshold:.4f}
                ''')

    title = ['Index', 'Patient', 'Progression-Free Survival', 'Progression-Free Survival Event', 
             'Overall Survival', 'Overall Survival Event', 'Survival Risk Score', 'Risk Level']
    pred_df = pd.DataFrame(index=[], columns=title)

    pred_list = []
    true_list = []

    start_time = time.time()

    model.eval()     

    with torch.inference_mode():
        with tqdm(total=len(test_loader), desc='Test', unit='iteration', ncols=100) as pbar:

            for idx, batch in enumerate(test_loader):

                image, clinic = batch['image'], batch['clinic']

                image = image.to(device=device, dtype=torch.float32)
                clinic = clinic.to(device=device, dtype=torch.long)

                seg_mask = seg_patient_image(segmodel, image)
                max_image, max_mask = find_max_tumor(image, seg_mask)   
                tumorimg, _ = split_tumor(max_image, max_mask)

                outputs = model(tumorimg)                                 

                clinic = clinic.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()

                pred_list.append(outputs)
                true_list.append(clinic)

                pbar.update(1)                                         

                patient_name_list = batch['name']                      

                for i in range(len(outputs)):

                    idx += 1                                         
                    patient_name = patient_name_list[i]               
                    patient_pfs = clinic[i][0]                               
                    patient_pfs_e = clinic[i][1]                    
                    patient_os = clinic[i][2]                        
                    patient_os_e = clinic[i][3]                      
                    patient_srs = outputs[i][0]                     

                    if patient_srs > threshold:
                        patient_rl = 'high risk'
                    else:
                        patient_rl = 'low risk'

                    tmp = pd.Series([idx, patient_name, patient_pfs, patient_pfs_e, patient_os, patient_os_e,
                                     patient_srs, patient_rl], index=title)
                    pred_df = pred_df._append(tmp, ignore_index=True)                    
                    pred_df.to_csv(outputs_path + 'risk_score_predict_table.csv', index=False)     

    end_time = time.time()

    pred_array = np.array(np.concatenate(pred_list))          
    true_array = np.array(np.concatenate(true_list))          

    pfs_ci = compute_concordance_index(true_array[:, 0:2], pred_array)      
    pfs_ci_95_low, pfs_ci_95_upp = compute_concordance_index_bootstrap_ci(true_array[:, 0:2], pred_array)
    os_ci = compute_concordance_index(true_array[:, 2:4], pred_array)       
    os_ci_95_low, os_ci_95_upp = compute_concordance_index_bootstrap_ci(true_array[:, 2:4], pred_array)

    logging.info(f'Test Progression-Free Survival CI: {pfs_ci:.4f} (95% CI:{pfs_ci_95_low:.4f}-{pfs_ci_95_upp:.4f}) \t' 
                 f'Test Overall Survival CI: {os_ci:.4f} (95% CI:{os_ci_95_low:.4f}-{os_ci_95_upp:.4f})')
    
    logging.info(f'Testing time: '
                 f'{int((end_time - start_time) / 3600)} hours ' 
                 f'{int(((end_time - start_time) % 3600) / 60)} minutes '
                 f'{int(((end_time - start_time) % 3600) % 60)} seconds ')


if __name__ == '__main__':

    data_path = ['clinical data directory...', 'image data directory...']

    logginglog_path = args.record_save_path + 'logginglogs/'
    os.makedirs(logginglog_path, exist_ok=True)
    logging.basicConfig(filename=logginglog_path + 'testlog.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('*********************************************************************************')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    pred_save_file = args.output_save_path
    os.makedirs(pred_save_file, exist_ok=True)

    model = predict_net(inchannel=3, outchannel=3, transformer_scare='base', is_pretrained=True)
    model.model_transformer.model.blocks = nn.Sequential(*list(model.model_transformer.model.blocks.children())[:-3])
    model.to(device=device)

    model_path = args.model_save_path + f'checkpoint_epoch{args.epochs}.pth'
    state_dict = torch.load(model_path, map_location=device)      
    model.load_state_dict(state_dict)                               
    logging.info(f'Loading model {model_path}')

    segmodel = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)                 
    segmodel.to(device=device)
    model_path = './seg_model/models/model/epoch_100.pth'
    segmodel.load_state_dict(torch.load(model_path, map_location=device))

    threshold = get_mid_risk_score(model, segmodel, device, data_path)

    predict(model, segmodel, device, data_path, pred_save_file, threshold)

    logging.info('*********************************************************************************\n\n')




