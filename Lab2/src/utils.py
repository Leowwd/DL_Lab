import torch

def calculate_dice_score(pred, target, smooth=1e-5):

    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    pred_size = pred_flat.sum()
    target_size = target_flat.sum()
    
    dice = (2. * intersection + smooth) / (pred_size + target_size + smooth)
    
    return dice.item()