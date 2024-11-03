import torch
import torch.nn.functional as F

from args import args


def find_max_tumor(image, mask):    

    batch_max_image = []                             
    batch_max_mask = []                               

    for ibatch in range(mask.shape[0]):            

        item_image = image[ibatch]                   
        item_mask = mask[ibatch]                     
        tumor_region_dict = {}                           

        for idx in range(1, item_mask.shape[0] - 1):       

            mask_count = torch.count_nonzero(item_mask[idx])
            tumor_region_dict.update({str(idx): mask_count})
        
        if len(tumor_region_dict) != 0:
            max_index = max(tumor_region_dict, key=tumor_region_dict.get)   
        else:
            max_index = int(item_mask.shape[0] / 2)

        max_image = item_image[int(max_index):int(max_index)+1]
        max_mask = item_mask[int(max_index):int(max_index)+1]
        max_image = torch.concat((max_image, max_image, max_image), axis=0)
        max_mask = torch.concat((max_mask, max_mask, max_mask), axis=0)
        batch_max_image.append(max_image)
        batch_max_mask.append(max_mask)

    batch_max_image_tensor = torch.stack(batch_max_image)
    batch_max_mask_tensor = torch.stack(batch_max_mask)
    
    return batch_max_image_tensor, batch_max_mask_tensor


def split_tumor(tumor_image, tumor_label):

    largest_tumor_image_region = []
    largest_tumor_label_region = []

    for i in range(tumor_label.shape[0]):

        x, y = torch.nonzero(tumor_label[i][1], as_tuple=True)

        h, w = tumor_label[i][1].shape[-2:]

        extended_pixels = args.extended_pixels

        if len(x) > 0 and len(y) > 0:
            x_min = max(0, int(torch.min(x)) - extended_pixels)
            x_max = min(w, int(torch.max(x)) + extended_pixels)
            y_min = max(0, int(torch.min(y)) - extended_pixels)
            y_max = min(h, int(torch.max(y)) + extended_pixels)
            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                x_min = 0
                x_max = w
                y_min = 0
                y_max = h
        else:
            x_min = 0
            x_max = w
            y_min = 0
            y_max = h

        channel_tumor_image_region = []
        channel_tumor_label_region = []

        for j in range(tumor_label.shape[1]):

            tumor_image_region = tumor_image[i][j][x_min:x_max, y_min:y_max]
            tumor_label_region = tumor_label[i][j][x_min:x_max, y_min:y_max]

            img_size = args.img_size

            if tumor_image_region.shape[-1] != img_size or tumor_image_region.shape[-2] != img_size:
                tumor_image_region = F.interpolate(tumor_image_region.unsqueeze(0).unsqueeze(0).float(), 
                                                size=(img_size, img_size), mode='bicubic')
                tumor_label_region = F.interpolate(tumor_label_region.unsqueeze(0).unsqueeze(0).float(), 
                                                size=(img_size, img_size), mode='nearest')
                channel_tumor_image_region.append(tumor_image_region.squeeze())
                channel_tumor_label_region.append(tumor_label_region.squeeze())
            else:
                channel_tumor_image_region.append(tumor_image_region)
                channel_tumor_label_region.append(tumor_label_region)
        
        largest_tumor_image_region.append(torch.stack(channel_tumor_image_region))
        largest_tumor_label_region.append(torch.stack(channel_tumor_label_region))

    largest_tumor_image_region = torch.stack(largest_tumor_image_region)
    largest_tumor_label_region = torch.stack(largest_tumor_label_region)

    return largest_tumor_image_region, largest_tumor_label_region



