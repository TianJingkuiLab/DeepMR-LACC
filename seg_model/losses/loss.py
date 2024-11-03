import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args


def dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)                         
    a = input.sum(dim=sum_dim)
    b = target.sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)          

    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):

    flat_input = input.flatten(0, 1)
    flat_target = target.flatten(0, 1)

    average_dice = dice_coeff(flat_input, flat_target, reduce_batch_first, epsilon)

    return average_dice


def dice_loss(input, target, multiclass=False):

    fn = multiclass_dice_coeff if multiclass else dice_coeff

    dice_coefficient = fn(input, target, reduce_batch_first=True)

    dice_loss = 1 - dice_coefficient

    return dice_loss


def total_loss(pred, truth):                                  

    ce_loss = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()

    if args.num_classes == 1:
        pred_label = pred.float()
        true_label = truth.float()

        pred_label = pred_label.squeeze(1)

        loss_ce = ce_loss(pred_label, true_label)

        s_pred_label = torch.sigmoid(pred_label)

        loss_dice = dice_loss(s_pred_label, true_label, multiclass=False)

        loss = loss_ce + loss_dice 

    else:
        pred_label = pred.float()
        true_label = truth.long()

        loss_ce = ce_loss(pred_label, true_label)

        s_pred_label = F.softmax(pred_label, dim=1)
        oh_true_label = F.one_hot(true_label, args.num_classes).permute(0, 3, 1, 2)

        loss_dice = dice_loss(s_pred_label, oh_true_label, multiclass=True)

        loss = loss_ce + loss_dice

    return loss, loss_ce, loss_dice
