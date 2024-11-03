import os
import sys
import glob
import torch
import random
import imageio
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor

from args import args
from dataset import ReadData
from losses.loss import total_loss as loss
from networks.unet_model import UNet, OutConv
from utils import evaluate, dice_coeff, multiclass_dice_coeff


def worker_init_fn(worker_id):      
    random.seed(args.seed + worker_id)


def unique_mask_values(idx, mask_dir):                        

    mask_file = glob.glob(mask_dir + idx)                
    mask = np.asarray(imageio.imread(mask_file[0]))        

    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])                
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def kf_train(model, model_path, device, mask_value, model_load_path):

    logginglog_path = model_path + "/logginglogs/"
    os.makedirs(logginglog_path, exist_ok=True)

    logging.basicConfig(filename=logginglog_path + "train&validatelog.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("\n*******************************************************************************")
    logging.info(str(args))

    logging.info(f'''
                 Epochs:          {args.max_epochs}
                 Device:          {device.type}
                 Images size:     {args.img_size}
                 num_classes:     {args.num_classes}
                 mask values:     {mask_value}
               loading model:     {model_load_path}
                 ''')
    
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    dataset = ReadData(data_dir=args.data_path, split="train data", mask_values=mask_value, crop_log=False, aug_log=True)             

    logging.info("The length of dataset is: {}".format(len(dataset)))              

    if args.n_gpu > 1: 
        model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)   

    max_epoch = args.max_epochs                                          
    max_fold = args.num_folds                                            

    model.train()                                                                                                            
    step = 0

    plot_loss = [[], [], []]
    plot_dice = [[], []]

    for epoch_num in range(1, max_epoch + 1):

        tr_fold_dice_score_list = []
        va_fold_dice_score_list = []

        fold_total_loss_list = []
        fold_ce_loss_list = []
        fold_dice_loss_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

            total_loss_list = []
            ce_loss_list = []
            dice_loss_list = []

            dice_socre_list = []

            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)

            trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, 
                                     pin_memory=True, worker_init_fn=worker_init_fn)
            valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8,
                                   pin_memory=True, worker_init_fn=worker_init_fn)

            with tqdm(total=len(trainloader), desc=f'Epoch:{epoch_num}/{max_epoch},Fold:{fold + 1}/{max_fold}', unit='iteration') as pbar:

                for _, sampled_batch in enumerate(trainloader):

                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    
                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)     

                    outputs = model(image_batch.float())                                       

                    total_loss, loss_ce, loss_dice = loss(outputs, label_batch)       

                    if args.num_classes == 1:

                        mask_pred = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)

                        score1 = dice_coeff(mask_pred, label_batch, reduce_batch_first=False)
                        dice_socre_list.append(score1)

                    else:
                        mask_pred = torch.argmax(outputs, dim=1)

                        mask_true = F.one_hot(label_batch.long(), args.num_classes).permute(0, 3, 1, 2).float()

                        mask_pred = F.one_hot(mask_pred, args.num_classes).permute(0, 3, 1, 2).float()

                        score1, _ = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                        dice_socre_list.append(score1)

                    optimizer.zero_grad()               
                    total_loss.backward()               
                    optimizer.step()                    

                    pbar.set_postfix(**{'total_loss': total_loss.item()})  

                    total_loss_list.append(total_loss.item())
                    ce_loss_list.append(loss_ce.item())
                    dice_loss_list.append(loss_dice.item())

                    pbar.update(1)                                                                    

            fold_total_loss_list.append(sum(total_loss_list) / len(total_loss_list))
            fold_ce_loss_list.append(sum(ce_loss_list) / len(ce_loss_list))
            fold_dice_loss_list.append(sum(dice_loss_list) / len(dice_loss_list))

            tr_fold_dice_score_list.append(sum(dice_socre_list) / len(dice_socre_list))

            val_metrics = evaluate(model, valloader, device)           

            va_fold_dice_score_list.append(val_metrics['dice'])

        if epoch_num > (max_epoch / 2):
            lr_ = args.base_lr * (1.0 - step / (max_epoch * max_fold)) ** 0.9
            for param_group in optimizer.param_groups:                                  
                param_group['lr'] = lr_                                        
                step = step + 1
        else:
            lr_ = args.base_lr

        avg_epoch_total_loss = sum(fold_total_loss_list) / len(fold_total_loss_list)
        avg_epoch_ce_loss = sum(fold_ce_loss_list) / len(fold_ce_loss_list)
        avg_epoch_dice_loss = sum(fold_dice_loss_list) / len(fold_dice_loss_list)
        
        plot_loss[0].append(avg_epoch_total_loss)
        plot_loss[1].append(avg_epoch_ce_loss)
        plot_loss[2].append(avg_epoch_dice_loss)

        logging.info('epoch_ce_loss: {:.4f} \t epoch_dice_loss: {:.4f} \t epoch_total_loss: {:.4f}'\
             .format(avg_epoch_ce_loss, avg_epoch_dice_loss, avg_epoch_total_loss))
        
        logging.info(f'current learning rate: {lr_}')

        tr_avg_dice = sum(tr_fold_dice_score_list) / len(tr_fold_dice_score_list)
        va_avg_dice = sum(va_fold_dice_score_list) / len(va_fold_dice_score_list)

        plot_dice[0].append(tr_avg_dice.item())
        plot_dice[1].append(va_avg_dice.item())


        logging.info('Train DICE score: {:.4f}'.format(tr_avg_dice))
        logging.info('Validate DICE score: {:.4f}'.format(va_avg_dice))
        if (epoch_num % 5) == 0:
            save_mode_path = os.path.join(model_path, '/model/epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        fig1, ax1 = plt.subplots()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.5)  
        x = range(1, epoch_num + 1)                              
        y = plot_loss
        ax1.set_title('Train loss')
        ax1.plot(x, y[0], 'r-', label='total loss')
        ax1.plot(x, y[1], 'g-', label='ce loss')
        ax1.plot(x, y[2], 'b-', label='dice loss')
        ax1.legend()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('score')
        plt.savefig(model_path + '/plot_img/train_losses.png')
        plt.close()

        fig2, ax2 = plt.subplots()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.5)  
        x = range(1, epoch_num + 1)                             
        z1 = plot_dice
        ax2.set_title('Dice score')
        ax2.plot(x, z1[0], 'r-', label='train dice')
        ax2.plot(x, z1[1], 'b-', label='validate dice')
        ax2.legend()
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('score')
        plt.savefig(model_path + '/plot_img/train&validate_dice.png')
        plt.close()

    print("Training Finished!")
    logging.info("\n*******************************************************************************")


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.deterministic:
        cudnn.benchmark = True                      
        cudnn.deterministic = False                 
    else:
        cudnn.benchmark = False                    
        cudnn.deterministic = True                 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_save_path = './models'  
    os.makedirs(model_save_path, exist_ok=True)                                                   

    label_dir = args.data_path + 'pretrained dataset label path...'
    image_label_name_list = os.listdir(label_dir)

    with ThreadPoolExecutor(max_workers=8) as executor:  
        unique = list(tqdm(executor.map(lambda x: unique_mask_values(x, mask_dir=label_dir), image_label_name_list), 
                    total=len(image_label_name_list)))
    mask_value = list(sorted(np.unique(np.concatenate(unique))))

    args.num_classes = len(mask_value) if len(mask_value) > 2 else 1

    net = UNet(n_channels=1, n_classes=args.num_classes, bilinear=args.bilinear)               

    model_path = model_save_path + '/pretrained_models/Unet_Synapse_Abdomen_B24_E100_224.pth'
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)

    args.num_classes = 1
    net.num_classes = 1

    net.outc = (OutConv(64, args.num_classes))

    args.data_path = 'fine tune dataset directory...'

    label_dir = args.data_path + 'fine tune dataset label path...'
    image_label_name_list = os.listdir(label_dir)

    with ThreadPoolExecutor(max_workers=8) as executor:
        unique = list(tqdm(executor.map(lambda x: unique_mask_values(x, mask_dir=label_dir), image_label_name_list), 
                    total=len(image_label_name_list)))
    mask_value = list(sorted(np.unique(np.concatenate(unique))))

    net = net.to(device=device)

    kf_train(net, model_save_path, device, mask_value, model_path)


