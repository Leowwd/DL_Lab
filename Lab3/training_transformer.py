import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        self.optim.zero_grad()
        for i, images in enumerate(pbar):
            images = images.to(args.device)
            logits, z_indices = self.model(images)

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            loss = loss / args.accum_grad
            loss.backward()
            
            if (i + 1) % args.accum_grad == 0 or (i + 1) == len(train_loader):
                self.optim.step()
                self.optim.zero_grad()
                
            total_loss += loss.item() * args.accum_grad
            pbar.set_postfix({'loss': f'{loss.item() * args.accum_grad:.4f}'})
            
        self.scheduler.step()
            
        return total_loss / len(train_loader)

    @torch.no_grad()
    def eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}")
        for i, images in enumerate(pbar):
            images = images.to(args.device)
            
            logits, z_indices = self.model(images)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(val_loader)

    def configure_optimizers(self):
        lr = args.learning_rate if args.learning_rate > 0 else 1e-4
        optimizer = torch.optim.AdamW(self.model.transformer.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch)
        
        if train_transformer.scheduler is not None:
            train_transformer.scheduler.step()
            current_lr = train_transformer.scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        else:
            print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if epoch % args.save_per_epoch == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(train_transformer.model.transformer.state_dict(), args.checkpoint_path)
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/epoch_{epoch}.pt")
            print(f"Model saved at epoch {epoch}!")