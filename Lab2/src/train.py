import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from oxford_pet import OxfordPetDataset
from models.resnet34_unet import ResNet34_UNet
from models.unet import UNet
from utils import calculate_dice_score
import argparse

def train(args):
    data_root = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
            ),
            A.RandomBrightnessContrast(p=0.2),
        ])
    
    valid_transform = A.Compose([
            A.Resize(256, 256),
        ])
    
    # You can modify the txt paths if needed, but they should be correct as is
    train_dataset = OxfordPetDataset(root_dir=data_root, txt_path=args.train_txt, mode="train", transform=train_transform)
    valid_dataset = OxfordPetDataset(root_dir=data_root, txt_path=args.val_txt, mode="valid", transform=valid_transform)
    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(valid_dataset, batch_size=16)

    model = ResNet34_UNet().to(device) if args.model == "res_unet" else UNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    os.makedirs("../saved_models", exist_ok=True)
    best_dice_score = 0.0
    epochs = args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)        
            loss = criterion(outputs, masks)
            total_loss += loss.item()        
            
            loss.backward()        
            optimizer.step()
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        total_val_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                total_val_dice += calculate_dice_score(preds, masks)
                
        avg_val_dice = total_val_dice / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        if avg_val_dice > best_dice_score:
            best_dice_score = avg_val_dice
            torch.save(model.state_dict(), args.model_path)
            
        scheduler.step()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../dataset/oxford-iiit-pet", help="Path to the dataset")
    parser.add_argument("--model", type=str, default="res_unet", choices=["unet", "res_unet"], help="Model architecture to use")
    parser.add_argument("--train_txt", default="../dataset/oxford-iiit-pet/train.txt", help="Path to train.txt")
    parser.add_argument("--val_txt", default="../dataset/oxford-iiit-pet/val.txt", help="Path to val.txt")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--model_path", default="../saved_models/best_RES_UNET.pth", help="Path to the saved model weights")
    args = parser.parse_args()
    train(args)