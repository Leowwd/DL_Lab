import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetDataset
from models.resnet34_unet import ResNet34_UNet
from models.unet import UNet
from utils import calculate_dice_score

def train():

    data_root = "../dataset/oxford-iiit-pet"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = OxfordPetDataset(root_dir=data_root, txt_name='train.txt', mode='train')
    valid_dataset = OxfordPetDataset(root_dir=data_root, txt_name='val.txt', mode='valid')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    model = UNet().to(device)

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("../saved_models", exist_ok=True)
    best_dice_score = 0.0
    epochs = 10

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
            torch.save(model.state_dict(), "../saved_models/best_UNet.pth")
            
if __name__ == "__main__":
    train()