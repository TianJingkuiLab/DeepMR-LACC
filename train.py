import os
import sys
import time
import glob
import torch
import random
import logging
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch import optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from args import args
from losses import cox_loss
from evaluate import evaluate
from dataset import ReadDataset
from networks.network import predict_net
from seg_model.networks.unet_model import UNet
from utils.c_index import compute_concordance_index
from utils.seg_patient_image import seg_patient_image
from utils.split_tumor import split_tumor, find_max_tumor


def train_model(model, segmodel, device, data_path, epochs, batch_size, learning_rate):

    tr_dataset = ReadDataset(data_path[1] + 'train data/', data_path[0], rot_log=True)
    val_dataset = ReadDataset(data_path[1] + 'val data/', data_path[0], rot_log=False)

    train_loader = DataLoader(tr_dataset, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    validate_loader = DataLoader(val_dataset, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Train size:      {len(tr_dataset)}
            Validate size:   {len(val_dataset)}
            Checkpoints:     {args.model_save_path}
            Device:          {device.type}
            ''')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    title = ['Epoch', 'Progression-Free Survival CI', 'Overall Survival CI']
    train_data_df = pd.DataFrame(index=[], columns=title)                                                  
 
    epoch_loss_list = [[], []]
    metrics_list = [[[], []], [[], []]]

    best_va_os_ci = 0
    index = 0
    current_lr = learning_rate

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        model.train()                                                
        pred_list = []
        true_list = []
        epoch_loss = []

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='iteration', ncols=100) as pbar:
            
            for _, batch in enumerate(train_loader):

                image, clinic = batch['image'], batch['clinic']

                image = image.to(device=device, dtype=torch.float32)
                clinic = clinic.to(device=device, dtype=torch.long)

                seg_mask = seg_patient_image(segmodel, image)
                max_image, max_mask = find_max_tumor(image, seg_mask)    
                tumorimg, _ = split_tumor(max_image, max_mask)

                outputs = model(tumorimg)                             

                pfs_loss = cox_loss(clinic[:, 0:2], outputs)                                
                os_loss = cox_loss(clinic[:, 2:4], outputs)                                 
                loss = args.loss_weight[0] * os_loss + args.loss_weight[1] * pfs_loss       

                pred_list.append(outputs.detach().cpu().numpy())
                true_list.append(clinic.detach().cpu().numpy())

                epoch_loss.append(loss.item())

                loss.requires_grad_(True)                           

                optimizer.zero_grad()                            
                loss.backward()                                  
                optimizer.step()                                  

                pbar.set_postfix(**{'loss': loss.item()})         
                
                pbar.update(1)                                  

        tr_epoch_loss = sum(epoch_loss) / len(epoch_loss)
 
        pred_array = np.array(np.concatenate(pred_list))       
        true_array = np.array(np.concatenate(true_list))

        tr_pfs_ci = compute_concordance_index(true_array[:, 0:2], pred_array)     
        tr_os_ci = compute_concordance_index(true_array[:, 2:4], pred_array)  

        tr_metrics = {'loss': tr_epoch_loss, 'pfs_ci': tr_pfs_ci, 'os_ci': tr_os_ci}    

        va_metrics = evaluate(model, segmodel, validate_loader, device)      
     
        scheduler.step()
        for param_group in optimizer.param_groups:                             
                current_lr = param_group['lr']

        temp = pd.Series([epoch, tr_metrics['pfs_ci'], tr_metrics['os_ci']], index=title)
        train_data_df = train_data_df._append(temp, ignore_index=True)
        train_data_df.to_csv(args.output_save_path + 'epoch_train_result_table.csv', index=False)

        logging.info(f"Epoch train loss: {tr_metrics['loss']:.4f} \t Epoch validate loss: {va_metrics['loss']:.4f} \t" 
                     f"Current learning rate: {current_lr}")

        logging.info(f"Train Progression-Free Survival CI: {tr_metrics['pfs_ci']:.4f}  \t Train Overall Survival CI: {tr_metrics['os_ci']:.4f} \n"
                     f"Validate Progression-Free Survival CI: {va_metrics['pfs_ci']:.4f} \t Validate Overall Survival CI: {va_metrics['os_ci']:.4f}")

        epoch_loss_list[0].append(tr_metrics['loss'])
        epoch_loss_list[1].append(va_metrics['loss'])

        metrics_list[0][0].append(tr_metrics['os_ci'])
        metrics_list[0][1].append(va_metrics['os_ci'])
        metrics_list[1][0].append(tr_metrics['pfs_ci'])
        metrics_list[1][1].append(va_metrics['pfs_ci'])

        model_save_path = args.model_save_path
        os.makedirs(model_save_path, exist_ok=True)   

        if (epoch % 1) == 0:
            state_dict = model.state_dict()                                       
            torch.save(state_dict, str(model_save_path + 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        if best_va_os_ci < (va_metrics['os_ci'] + va_metrics['pfs_ci']):
            best_va_os_ci = va_metrics['os_ci'] + va_metrics['pfs_ci']
            best_state_dict =  model.state_dict() 
            index = epoch
  
        plt_save_path = args.visualization_save_path + 'train&validate/'
        os.makedirs(plt_save_path, exist_ok=True)

        fig, ax = plt.subplots()
        x = range(1, epoch + 1)                                  
        y1 = epoch_loss_list[0]
        y2 = epoch_loss_list[1]
        fig.suptitle('Train&Validate Loss')
        ax.plot(x, y1, 'r-', label='train loss')
        ax.plot(x, y2, 'b-', label='validate loss')
        ax.legend()
        ax.set_xlabel('epoch')
        plt.savefig(plt_save_path + 'train&validate_loss.png')
        plt.close()

        fig, ax = plt.subplots()
        x = range(1, epoch + 1)
        y = metrics_list[0]
        fig.suptitle('Train&Validate Overall Survival Concordance Index')
        ax.plot(x, y[0], 'r-', label='train os ci')
        ax.plot(x, y[1], 'b-', label='validate os ci')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('score')
        plt.savefig(plt_save_path + 'train&validate_os_ci.png')
        plt.close()

        fig, ax = plt.subplots()
        x = range(1, epoch + 1)
        y = metrics_list[1]
        fig.suptitle('Train&Validate Progression-Free Survival Concordance Index')
        ax.plot(x, y[0], 'r-', label='train pfs ci')
        ax.plot(x, y[1], 'b-', label='validate pfs ci')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('score')
        plt.savefig(plt_save_path + 'train&validate_pfs_ci.png')
        plt.close()
    
    end_time = time.time()

    if glob.glob(model_save_path + 'best_checkpoint*') != []:
        os.remove(glob.glob(model_save_path + 'best_checkpoint_epoch*')[0])
    torch.save(best_state_dict, str(model_save_path + 'best_checkpoint_epoch{}.pth'.format(index)))
    logging.info(f'Best Checkpoint {index} saved!')

    if os.path.exists(args.output_save_path + 'train_risk_score_table.csv'):
        os.remove(args.output_save_path + 'train_risk_score_table.csv')

    logging.info(f'Training time: '
                 f'{int((end_time - start_time) / 3600)} hours ' 
                 f'{int(((end_time - start_time) % 3600) / 60)} minutes '
                 f'{int(((end_time - start_time) % 3600) % 60)} seconds ')

 
if __name__ == "__main__":

    data_path = ['clinical data directory...', 'image data directory...']
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logginglog_path = args.record_save_path + '/logginglogs/'
    os.makedirs(logginglog_path, exist_ok=True)

    logging.basicConfig(filename=logginglog_path + 'train&validatelog.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('*********************************************************************************')

    model = predict_net(inchannel=3, outchannel=3, transformer_scare='base', is_pretrained=True)
    model.model_transformer.model.blocks = nn.Sequential(*list(model.model_transformer.model.blocks.children())[:-3])      
    model.to(device=device)

    segmodel = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)                
    segmodel.to(device=device)
    model_path = './seg_model/models/model/epoch_100.pth'
    segmodel.load_state_dict(torch.load(model_path, map_location=device))

    train_model(model=model, segmodel=segmodel, device=device, data_path=data_path, epochs=args.epochs,
                   batch_size=args.batch_size, learning_rate=args.learning_rate)

    logging.info('*********************************************************************************\n\n')





