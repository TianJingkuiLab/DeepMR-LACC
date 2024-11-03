"""from https://github.com/jacobgil/vit-explain?tab=readme-ov-file"""

import os
import cv2
import sys
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from args import args
from dataset import ReadDataset
from networks.network import predict_net
from seg_model.networks.unet_model import UNet
from utils.seg_patient_image import seg_patient_image
from utils.split_tumor import split_tumor, find_max_tumor


def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)

            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)
    
    mask = result[0, 0 , 1 :]

    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    


class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    mask = result[0, 0 , 1 :]

    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * 0.5 + np.float32(img)
    cam = cam / np.max(cam)
    return heatmap, np.uint8(255 * cam)


if __name__ == '__main__':

    data_path = ['clinical data directory...', 'image data directory...']

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = predict_net(inchannel=3, outchannel=3, transformer_scare='base', is_pretrained=True)
    model.model_transformer.model.blocks = nn.Sequential(*list(model.model_transformer.model.blocks.children())[:-3])
    model.to(device=device)

    model_path = args.model_save_path + f'checkpoint_epoch{args.epochs}.pth'
    state_dict = torch.load(model_path)       
    model.load_state_dict(state_dict) 

    segmodel = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)                 
    segmodel.to(device=device)
    model_path = './seg_model/models/model/epoch_100.pth'
    segmodel.load_state_dict(torch.load(model_path))

    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
    test_dataset = ReadDataset(data_path[1] + 'test data/', data_path[0], rot_log=False)
    test_loader = DataLoader(test_dataset, shuffle=True, drop_last=True, **loader_args)

    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=100):

        image = batch['image']
        image = image.to(device=device, dtype=torch.float32)

        seg_mask = seg_patient_image(segmodel, image)
        max_image, max_mask = find_max_tumor(image, seg_mask)
        tumorimg, tumormask = split_tumor(max_image, max_mask)

        output = model(tumorimg)

        category_index = None
        head_fusion = 'max'
        discard_ratio = 0.9
        if category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=head_fusion, discard_ratio=discard_ratio)
            mask = attention_rollout(tumorimg)
            name = "attention_rollout_{:.3f}_{}.png".format(discard_ratio, head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=discard_ratio)
            mask = grad_rollout(tumorimg, category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(category_index, discard_ratio, head_fusion)

        ori_img = tumorimg
        np_img = ori_img.squeeze().cpu().detach().numpy() * 255
        np_img = np_img.astype(np.uint8).transpose((1, 2, 0))
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        heatmap, mask = show_mask_on_image(np_img, mask)

        b, g, r = cv2.split(heatmap)
        heatmap = cv2.merge((r, g, b))
        b, g, r = cv2.split(mask)
        mask = cv2.merge((r, g, b))

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(np_img)
        ax[0].axis('off')

        ax[1].imshow(mask)
        ax[1].axis('off')

        plt.savefig(args.visualization_save_path + f'heatmap/Attention Rollout/{batch["name"][0]}.png')
        plt.close()



