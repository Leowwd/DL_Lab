import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = torch.mean((imgs1 - imgs2) ** 2, dim=[1, 2, 3])
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return torch.mean(psnr)


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.n_epoch = args.num_epoch
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        
        if self.kl_anneal_type == 'Cyclical':
            self.beta_list = self.frange_cycle_linear(
                self.n_epoch, start=0.0, stop=1.0,
                n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio
            )
        elif self.kl_anneal_type == 'Monotonic':
            self.beta_list = self.frange_cycle_linear(
                self.n_epoch, start=0.0, stop=1.0,
                n_cycle=1, ratio=self.kl_anneal_ratio
            )
        else:  # Without KL annealing
            self.beta_list = [1.0] * self.n_epoch
        
    def update(self):
        self.current_epoch += 1
    
    def get_beta(self):
        if self.current_epoch < len(self.beta_list):
            return self.beta_list[self.current_epoch]
        return 1.0

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        """
        Cyclical annealing schedule from Fu et al. (NAACL 2019).
        n_iter: total number of iterations (epochs).
        n_cycle: number of cycles (M).
        ratio: proportion of each cycle used for increasing beta (R).
        """
        L = [stop] * n_iter
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio) if (period * ratio) > 0 else 0
        
        for c in range(n_cycle):
            v = start
            i = 0
            while v <= stop and int(i + c * period) < n_iter:
                L[int(i + c * period)] = v
                v += step
                i += 1
        
        return L
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr) if self.args.optim == "Adam" else optim.AdamW(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        best_val_psnr = -1.0  # PSNR is higher the better
        
        # Tracking metrics for the Report
        self.metrics = {
            "train_loss": [], "val_loss": [], "val_psnr": [], "tfr": [], "beta": []
        }
        
        import json
        
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            
            epoch_train_losses = []
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                adapt_TeacherForcing = True if random.random() < self.tfr else False
                
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_train_losses.append(loss.item())
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            val_loss, val_psnr = self.eval()
            
            # Save metrics to JSON
            self.metrics["train_loss"].append(sum(epoch_train_losses)/len(epoch_train_losses))
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_psnr"].append(val_psnr)
            self.metrics["tfr"].append(self.tfr)
            self.metrics["beta"].append(beta)
            
            with open(os.path.join(self.args.save_root, "metrics.json"), "w") as f:
                json.dump(self.metrics, f)
            
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                print(f"New best model! val_psnr={best_val_psnr:.4f} @ epoch {self.current_epoch} -> best_model.ckpt")
                self.save(os.path.join(self.args.save_root, "best_model.ckpt"))
                
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0.0
        total_psnr = 0.0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, avg_psnr, _ = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            total_loss += loss.item()
            total_psnr += avg_psnr
        avg_psnr_epoch = total_psnr / len(val_loader)
        print(f"[Epoch {self.current_epoch}] Avg Val PSNR: {avg_psnr_epoch:.4f} dB")
        return total_loss / len(val_loader), avg_psnr_epoch
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        img = img.permute(1, 0, 2, 3, 4)      # (B, T, C, H, W) -> (T, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W) -> (T, B, C, H, W)
        
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-1])    # flip W dimension
            label = torch.flip(label, dims=[-1])
        
        # ColorJitter on img ONLY (label colors are semantic, must not be changed)
        if random.random() > 0.5:
            # Same jitter params applied to ALL frames in the sequence
            brightness = 1.0 + random.uniform(-0.2, 0.2)
            contrast   = 1.0 + random.uniform(-0.2, 0.2)
            img = torch.clamp(img * brightness, 0, 1)
            img = torch.clamp((img - 0.5) * contrast + 0.5, 0, 1)
        
        seq_len = img.shape[0]
        mse_loss = 0
        kl_loss = 0
        
        # The last generated frame; starts as the first ground truth frame x_0
        last_frame = img[0]
        
        for t in range(1, seq_len):
            # Encode the ground truth current frame and label
            human_feat_gt = self.frame_transformation(img[t])
            label_feat = self.label_transformation(label[t])
            
            # Posterior predictor: encode (frame_feat, label_feat) -> z, mu, logvar
            z, mu, logvar = self.Gaussian_Predictor(human_feat_gt, label_feat)
            
            # Determine the previous frame to use as decoder input
            if adapt_TeacherForcing:
                # Teacher forcing: use ground truth previous frame
                prev_frame_feat = self.frame_transformation(img[t - 1])
            else:
                # Autoregressive: use last generated frame
                prev_frame_feat = self.frame_transformation(last_frame)
            
            # Decoder fusion: combine prev frame features, label features, and latent z
            decoded_feat = self.Decoder_Fusion(prev_frame_feat, label_feat, z)
            
            generated_frame = self.Generator(decoded_feat)
            
            # Update last frame for autoregressive mode
            last_frame = generated_frame
            
            # Accumulate losses
            mse_loss += self.mse_criterion(generated_frame, img[t])
            kl_loss += kl_criterion(mu, logvar, self.batch_size)
        
        beta = self.kl_annealing.get_beta()
        loss = mse_loss + beta * kl_loss
        
        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()
        
        return loss
    
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4)      # (B, T, C, H, W) -> (T, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W) -> (T, B, C, H, W)
        
        seq_len = img.shape[0]
        mse_loss = 0
        psnr_list = []  # collect per-frame PSNR
        
        # Start with the first ground truth frame
        last_frame = img[0]
        
        # store tensors on GPU (.detach() only, no .cpu() inside loop)
        decoded_frame_list = [img[0].detach()]
        
        for t in range(1, seq_len):
            label_feat = self.label_transformation(label[t])
            
            # Sample z from the prior distribution N(0, I)
            z = torch.randn(img.shape[1], self.args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)
            
            # Encode the previous frame (autoregressive)
            prev_frame_feat = self.frame_transformation(last_frame)
            
            # Decoder fusion
            decoded_feat = self.Decoder_Fusion(prev_frame_feat, label_feat, z)
            
            generated_frame = self.Generator(decoded_feat)
            last_frame = generated_frame
            
            mse_loss += self.mse_criterion(generated_frame, img[t])
            
            # Compute per-frame PSNR
            psnr_list.append(Generate_PSNR(generated_frame.clamp(0, 1), img[t]).item())
            
            # keep on GPU
            decoded_frame_list.append(generated_frame.detach())
        
        # Single bulk transfer to CPU after loop
        if self.args.store_visualization and self.current_epoch % 5 == 0:
            frames_cpu = stack(decoded_frame_list).permute(1, 0, 2, 3, 4).cpu()
            self.make_gif(frames_cpu[0], os.path.join(self.args.save_root,
                          f'epoch={self.current_epoch}_val.gif'))
        
        avg_psnr = sum(psnr_list) / len(psnr_list)
        return mse_loss, avg_psnr, psnr_list
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr) if self.args.optim == "Adam" else optim.AdamW(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
