import torch
import numpy as np
from tqdm import tqdm

from args import args
from losses import cox_loss
from utils.c_index import compute_concordance_index
from utils.seg_patient_image import seg_patient_image
from utils.split_tumor import split_tumor, find_max_tumor

                                  
def evaluate(net, segnet, dataloader, device):

    net.eval()  

    epoch_loss = []
    pred_list = []
    true_list = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader), desc='Validate', unit='iteration', leave=False, ncols=100):

            image, clinic = batch['image'], batch['clinic']

            image = image.to(device=device, dtype=torch.float32)
            clinic = clinic.to(device=device, dtype=torch.long)

            seg_mask = seg_patient_image(segnet, image)
            max_image, max_mask = find_max_tumor(image, seg_mask, batch['name'])   
            tumorimg, _ = split_tumor(max_image, max_mask)

            outputs = net(tumorimg)

            pfs_loss = cox_loss(clinic[:, 0:2], outputs)                             
            os_loss = cox_loss(clinic[:, 2:4], outputs)                              
            loss = args.loss_weight[0] * os_loss + args.loss_weight[1] * pfs_loss   

            epoch_loss.append(loss.item())

            pred_list.append(outputs.detach().cpu().numpy())
            true_list.append(clinic.detach().cpu().numpy())

    pred_array = np.array(np.concatenate(pred_list))                                 
    true_array = np.array(np.concatenate(true_list))                                 

    va_pfs_ci = compute_concordance_index(true_array[:, 0:2], pred_array)               
    va_os_ci = compute_concordance_index(true_array[:, 2:4], pred_array)                
 
    va_epoch_loss = sum(epoch_loss) / len(epoch_loss)                                       

    net.train()                                                                      

    evaluate_metrics = {'loss': va_epoch_loss, 'pfs_ci': va_pfs_ci, 'os_ci': va_os_ci}

    return evaluate_metrics




