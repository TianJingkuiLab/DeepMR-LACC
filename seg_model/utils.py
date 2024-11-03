import torch
from tqdm import tqdm
from args import args
import torch.nn.functional as F


def dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)                         
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)          

    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)

    return dice.mean()


def multiclass_dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):

    flat_input = input.flatten(0, 1)
    flat_target = target.flatten(0, 1)

    average_dice = dice_coeff(flat_input, flat_target, reduce_batch_first, epsilon)

    new_input = input.permute(1, 0, 2, 3)
    new_target = target.permute(1, 0, 2, 3)
    class_dice = []
    for i in range(new_input.shape[0]):
        dice = dice_coeff(new_input[i], new_target[i], reduce_batch_first, epsilon)
        class_dice.append(dice)

    return average_dice, class_dice


@torch.inference_mode()                                     
def evaluate(net, dataloader, device):
    net.eval()                                              
    num_val_batches = len(dataloader)                     
    dice_score = 0
    dice_score_list = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validate', unit='iteration', leave=False, ncols=100):

        image, mask_true = batch['image'], batch['label']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        mask_pred = net(image)

        if args.num_classes == 1:
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'

            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float().squeeze(1)

            score1 = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            dice_score = dice_score + score1

            if score1 > 1e-5:
                dice_score_list.append(score1)

        else:
            assert mask_true.min() >= 0 and mask_true.max() < args.num_classes, 'True mask indices should be in [0, n_classes]'

            mask_pred = torch.argmax(mask_pred, dim=1)

            mask_true = F.one_hot(mask_true.long(), args.num_classes).permute(0, 3, 1, 2).float()

            mask_pred = F.one_hot(mask_pred, args.num_classes).permute(0, 3, 1, 2).float()

            score1, _ = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            dice_score = dice_score + score1

            if score1 > 1e-5:
                dice_score_list.append(score1)

    avg_dice = dice_score / max(num_val_batches, 1)          

    metrics = {'dice': avg_dice, 'dice_list': dice_score_list}
    net.train()                                           

    return metrics

