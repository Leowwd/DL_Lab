import torch
from torch.utils.data import DataLoader
from oxford_pet import OxfordPetDataset
from models.unet import UNet
from utils import calculate_dice_score 

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_dataset = OxfordPetDataset(root_dir='../dataset/oxford-iiit-pet', txt_name="val.txt", mode='valid')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = UNet().to(device)
    model.load_state_dict(torch.load("saved_models/best_UNet.pth"))
    model.eval()
    
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            total_dice += calculate_dice_score(preds, masks)
            
    print(f"Dice Score: {total_dice / len(val_loader):.4f}")

if __name__ == '__main__':
    evaluate()