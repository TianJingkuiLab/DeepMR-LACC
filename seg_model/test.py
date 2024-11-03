import os
import sys
import glob
import torch
import random
import imageio
import logging
import numpy as np
from tqdm import tqdm
import seaborn as sns
from imageio import imsave
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

from args import args
from dataset import ReadData
from networks.unet_model import UNet
from utils import dice_coeff, multiclass_dice_coeff


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


def output_save(label_savepath, cmpimg_savepath, ori_img, pred_mask, true_mask, name):

    imsave(label_savepath + name + '.png', pred_mask)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title('Input image')
    ax[0, 0].imshow(ori_img, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 1].set_title('Predict label')
    ax[0, 1].imshow(pred_mask, cmap='gray')
    ax[0, 1].axis('off')
    ax[0, 2].set_title('True label')
    ax[0, 2].imshow(true_mask, cmap='gray')
    ax[0, 2].axis('off')
    ax[1, 0].set_title('Input image')
    ax[1, 0].imshow(ori_img, cmap='gray')
    ax[1, 0].axis('off')
    ax[1, 1].set_title('Pred mask in image')
    ax[1, 1].imshow(ori_img, cmap='gray')
    ax[1, 1].imshow(pred_mask, cmap='gray', alpha=0.3)
    ax[1, 1].axis('off')
    ax[1, 2].set_title('True mask in image')
    ax[1, 2].imshow(ori_img, cmap='gray')
    ax[1, 2].imshow(true_mask, cmap='gray', alpha=0.3)
    ax[1, 2].axis('off')
    plt.savefig(cmpimg_savepath + name + '.png')
    plt.close()


def test_single_volume(image, label, net, test_save_path=None, case=None):

    input = image.float().cuda()

    net.eval()
    with torch.no_grad():
        
        outputs = net(input)                 

        if args.num_classes > 1:
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            mask_true = F.one_hot(label.cuda().long(), args.num_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(out, args.num_classes).permute(0, 3, 1, 2).float()
            dice, _ = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

        else:
            out = (torch.sigmoid(outputs) > 0.5).float()
            dice = dice_coeff(out.squeeze(1), label.cuda(), reduce_batch_first=False)

    if test_save_path is not None:

        img = image.squeeze(1)
        img = ((img - img.min()) / (img.max() - img.min())) * 255.
        img = img.cpu().detach().numpy()
        img = img.astype(np.uint8)

        pred = out.squeeze(1).cpu().detach().numpy().astype(np.uint8) 
        mask = label.long().cpu().detach().numpy().astype(np.uint8) 

        path1 = test_save_path + '/pred_label/'
        path2 = test_save_path + '/image_label/'
        os.makedirs(path1, exist_ok=True)
        os.makedirs(path2, exist_ok=True)

        class_to_color = {
            0: [0, 0, 0],          
            1: [255, 255, 255],     
            2: [255, 0, 0],        
            3: [0, 255, 0],         
            4: [0, 0, 255],        
            5: [255, 255, 0],      
            6: [255, 0, 255],     
            7: [0, 255, 255],     
            8: [128, 128, 128],    
            9: [128, 0, 0],        
            10: [0, 128, 0],      
            11: [0, 0, 128],        
            12: [128, 128, 0],     
            13: [128, 0, 128],      
            14: [0, 128, 128],      
        }

        for idx in range(img.shape[0]):

            color_pred = np.zeros((pred[idx].shape[0], pred[idx].shape[1], 3), dtype=np.uint8)
            color_mask = np.zeros((mask[idx].shape[0], mask[idx].shape[1], 3), dtype=np.uint8)

            for class_idx, color in class_to_color.items():
                color_pred[pred[idx] == class_idx] = color
                color_mask[mask[idx] == class_idx] = color

            output_save(path1, path2, img[idx], color_pred, color_mask, case[idx])

    return dice


def inference(model, test_save_path, device, mask_value):

    model = model.to(device)

    test_dataset = ReadData(data_dir=args.data_path, split="test data", mask_values=mask_value, crop_log=False, aug_log=False)             
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    dataset_len = len(testloader)

    logging.info(f'''
                 Epochs:          {args.max_epochs}
                 Device:          {device.type}
                 Images size:     {args.img_size}
                 num classes:     {args.num_classes}
                 mask values:     {mask_value}
                 ''')

    logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()                                                                       

    dice_list = []

    for _, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), desc='Test', unit='iteration', leave=False, ncols=100):

        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']                                                     

        dice = test_single_volume(image, label, model, test_save_path=test_save_path, case=case_name)
        dice_list.append(dice)

    dice = np.mean(dice_list)

    logging.info('Dice Coefficient(Dice): {:.4f}'.format(dice))
    print("Testing Finished!")
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

    args.data_path = 'fine tune dataset directory..'

    label_dir = args.data_path + 'fine tune dataset label path...'
    image_label_name_list = os.listdir(label_dir)

    with ThreadPoolExecutor(max_workers=8) as executor: 
        unique = list(tqdm(executor.map(lambda x: unique_mask_values(x, mask_dir=label_dir), image_label_name_list), 
                    total=len(image_label_name_list)))
    mask_value = list(sorted(np.unique(np.concatenate(unique))))

    args.num_classes = len(mask_value) if len(mask_value) > 2 else 1

    net = UNet(n_channels=1, n_classes=args.num_classes, bilinear=args.bilinear)              

    model_path = './seg_model/models/model/epoch_' + str(args.max_epochs) + '.pth'

    net.load_state_dict(torch.load(model_path))

    logginglog_path = "./seg_model/models/logginglogs/"
    os.makedirs(logginglog_path, exist_ok=True)

    test_save_path = './seg_model/predictions'
    os.makedirs(test_save_path, exist_ok=True)

    logging.basicConfig(filename=logginglog_path + "testlog.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("\n*******************************************************************************")
    logging.info(str(args)) 

    inference(net, test_save_path, device, mask_value)
