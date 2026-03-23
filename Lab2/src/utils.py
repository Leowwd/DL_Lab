import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_dice_score(pred, target, smooth=1e-5):

    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    pred_size = pred_flat.sum()
    target_size = target_flat.sum()
    
    dice = (2. * intersection + smooth) / (pred_size + target_size + smooth)
    
    return dice.item()

def visualize_prediction(image, mask, pred, save_path=None):

    img_np = image.cpu().detach().numpy()
    mask_np = mask.cpu().detach().numpy().squeeze()
    pred_np = pred.cpu().detach().numpy().squeeze()
    
    img_np = np.transpose(img_np, (1, 2, 0))
    

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()