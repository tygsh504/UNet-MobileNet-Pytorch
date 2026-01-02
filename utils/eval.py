# import torch
# import torch.nn.functional as F
# from tqdm import tqdm

# from utils.dice_loss import dice_coeff


# def eval_net(net, loader, device):
#     """Evaluation without the densecrf with the dice coefficient"""
#     net.eval()
#     mask_type = torch.float32 if net.num_classes == 1 else torch.long
#     n_val = len(loader)  # the number of batch
#     tot = 0

#     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#         for batch in loader:
#             imgs, true_masks = batch['image'], batch['mask']
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             true_masks = true_masks.to(device=device, dtype=mask_type)

#             with torch.no_grad():
#                 mask_pred = net(imgs)

#             if net.num_classes > 1:
#                 tot += F.cross_entropy(mask_pred, true_masks).item()
#             else:
#                 pred = torch.sigmoid(mask_pred)
#                 pred = (pred > 0.5).float()
#                 tot += dice_coeff(pred, true_masks).item()
#             pbar.update()

#     net.train()
#     return tot / n_val

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_loss import dice_coeff

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.num_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot_loss = 0
    tot_dice = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.num_classes > 1:
                # Multiclass: CrossEntropy
                tot_loss += F.cross_entropy(mask_pred, true_masks).item()
                tot_dice += 0 # Placeholder for multiclass dice if needed
            else:
                # Binary: BCE Loss + Dice
                # 1. Calculate Loss
                tot_loss += F.binary_cross_entropy_with_logits(mask_pred, true_masks).item()
                
                # 2. Calculate Dice
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot_dice += dice_coeff(pred, true_masks).item()
            
            pbar.update()

    net.train()
    return tot_loss / n_val, tot_dice / n_val