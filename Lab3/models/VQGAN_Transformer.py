import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        _, indices, _ = self.vqgan.encode(x)
        indices = indices.view(x.shape[0], -1)
        return indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: math.cos(r * math.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        
        z_indices = self.encode_to_z(x) 
        B, N = z_indices.shape
        
        ratio = np.random.uniform(0, 1)
        mask_ratio = self.gamma(ratio)
        num_masked = math.floor(mask_ratio * N)
        
        mask = torch.zeros(B, N, device=x.device)
        rand = torch.rand(B, N, device=x.device)
        masked_indices = rand.topk(num_masked, dim=-1).indices
        mask.scatter_(1, masked_indices, 1)
        mask = mask.bool()
        
        masked_z = z_indices.clone()
        masked_z[mask] = self.mask_token_id
        logits = self.transformer(masked_z)
        
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, ratio):
        logits = self.transformer(z_indices)
        probs = torch.softmax(logits, dim=-1)
        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)

        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob) + 1e-9))
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        confidence[~mask] = float('inf')
        _, sorted_indices = torch.sort(confidence, dim=-1, descending=False)
        
        # define how much the iteration remain predicted tokens by mask scheduling
        mask_ratio = self.gamma(ratio)
        num_masked = math.floor(mask_ratio * self.num_image_tokens)
        
        mask_bc = torch.zeros_like(mask)
        for b in range(z_indices.shape[0]):
            mask_bc[b, sorted_indices[b, :num_masked]] = True
            
        ## At the end of the decoding process, add back the original(non-masked) token values
        z_indices_predict[~mask] = z_indices[~mask]
        
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
