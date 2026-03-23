import torch
from torch.utils.data import DataLoader
from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import calculate_dice_score, visualize_prediction
import argparse

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = OxfordPetDataset(root_dir="../dataset/oxford-iiit-pet", txt_name="val.txt", mode="valid")
    val_loader = DataLoader(val_dataset)
    
    model = ResNet34_UNet().to(device) if args.model == "res_unet" else UNet().to(device)
    model.load_state_dict(torch.load("../saved_models/best_" + args.model.upper() + ".pth"))
    model.eval()
    
    total_dice = 0.0
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            total_dice += calculate_dice_score(preds, masks)
            
            if i < 5:
                save_name = f"../saved_models/val_vis_{i}.png"
                visualize_prediction(images[0], masks[0], preds[0], save_path=save_name)
                print(f"Saved visualization to {save_name}")
            
    print(f"Dice Score: {total_dice / len(val_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="res_unet", choices=["unet", "res_unet"], help="Model architecture to use")
    args = parser.parse_args()
    evaluate(args)