import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from torch import nn, optim, utils
from torch.utils.data import Dataset, DataLoader, random_split
import opt_einsum
import torch.nn.functional as F
from pytorch_optimizer import SOAP

from Functions_and_classes import integrate, relative_l2_loss
    
def time_weighted_relative_loss(y_pred, y_true):

    # Norm at each timestep independently
    num = torch.norm(y_pred - y_true, p=2, dim=-1)   # (B, Nt)
    den = torch.norm(y_true,          p=2, dim=-1).clamp(min=1e-8)  # (B, Nt)
    
    per_t_loss = num / den        
    return per_t_loss.mean()

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
class VlasovDeepONet_CNN(L.LightningModule):
    def __init__(self, input_dim=4, Nv=32, Nt=16, basis_matrix=None):
        super().__init__()
        self.save_hyperparameters(ignore=['basis_matrix'])
        self.Nt = Nt 
        self.Nv = Nv
        
        if basis_matrix is None:
            raise ValueError("Must provide precomputed basis_matrix")
        self.K = basis_matrix.shape[0] 
        
        # Sanity check to ensure the decoder scales match the basis matrix
        assert self.K == 512, f"This decoder is hardcoded for K=512 (1024 total), but got K={self.K}"
        
        self.register_buffer('basis_real', basis_matrix)
        
        # input: params (4) + Flattened IC (Nv real + Nv imag)
        initial_size = self.Nv * 2
        total_input = input_dim + initial_size
        
        # 1D Decoder Branch
        self.branch = nn.Sequential(
            # 1. Project input (68) to bottleneck (512)
            nn.Linear(total_input, 512),
            nn.ReLU(),
            
            # 2. Reshape to (Channels, Length) -> (512, 1)
            Reshape(512, 1),
            
            # 3. Expand Length: 1 -> 8
            nn.ConvTranspose1d(512, 256, kernel_size=8, stride=8, bias=False),
            nn.ReLU(),
            
            # 4. Expand Length: 8 -> 32
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=4, bias=False),
            nn.ReLU(),
            
            # 5. Expand Length: 32 -> 128
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, bias=False),
            nn.ReLU(),
            
            # 6. Expand Length: 128 -> 512, Output 2 Channels (Real and Imaginary)
            nn.ConvTranspose1d(64, 2, kernel_size=4, stride=4, bias=False),
            
            # 7. Flatten (Batch, 2, 512) -> (Batch, 1024)
            # Channel 0 (Real) becomes the first 512 elements, Channel 1 (Imag) becomes the last 512.
            nn.Flatten()
        )

    def forward(self, params, initial_conditions):
        """
        Predicts exactly ONE block of time.
        initial_conditions shape: (Batch, Nv) complex
        """
        batch_size = params.shape[0]
        
        # 1. Prepare input for the Branch
        # Flatten and split complex into real/imag
        init_flat = initial_conditions.view(batch_size, -1)
        init_real_imag = torch.cat([init_flat.real, init_flat.imag], dim=1)
        
        x = torch.cat([params, init_real_imag], dim=1)
        
        # 2. Get Coefficients
        out = self.branch(x)
        c_real = out[..., :self.K]  # Slices the first 512 (Channel 0 from ConvTranspose)
        c_imag = out[..., self.K:]  # Slices the last 512 (Channel 1 from ConvTranspose)
        
        # 3. Reconstruct using Basis
        pred_real = c_real @ self.basis_real
        pred_imag = c_imag @ self.basis_real
        
        preds = torch.complex(pred_real, pred_imag)
        return preds.view(batch_size, self.Nt, self.Nv)


    def rollout(self, params, initial_conditions, num_blocks):
        batch_size = params.shape[0]
        current_ic = initial_conditions
        all_preds = []

        for block_idx in range(num_blocks):
            pred_flat = self(params, current_ic)
            pred_3d = pred_flat.view(batch_size, self.Nt, self.Nv)
            
            if block_idx == 0:
                all_preds.append(pred_3d)          # keep all Nt steps
            else:
                all_preds.append(pred_3d[:, 1:, :]) # skip the duplicated IC step
            
            current_ic = pred_3d[:, -1, :]

        return torch.cat(all_preds, dim=1)  # (B, Nt + (num_blocks-1)*(Nt-1), Nv)
    
    def training_step(self, batch, batch_idx):
        params, initial_conditions, solution = batch
        if solution.dim() == 2:
            solution = solution.view(solution.shape[0], -1, self.Nv)
        
        num_blocks = solution.shape[1] // self.Nt
        y_hat = self.rollout(params, initial_conditions, num_blocks)  # (B, total_Nt, Nv)
        
        # rel l2 loss
        loss_shape = time_weighted_relative_loss(y_hat, solution) 
        
        # Growth loss
        log_norm_pred = torch.log(torch.norm(y_hat,     p=2, dim=-1).clamp(min=1e-8))
        log_norm_true = torch.log(torch.norm(solution,  p=2, dim=-1).clamp(min=1e-8))
        loss_growth = F.mse_loss(log_norm_pred, log_norm_true)
        
        loss = loss_shape + 0.1 * loss_growth
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        params, initial_conditions, solution = batch
        num_blocks = solution.shape[1] // self.Nt
        y_hat = self.rollout(params, initial_conditions, num_blocks)
        
        val_loss = relative_l2_loss(y_hat, solution)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return SOAP(
        self.parameters(),
        lr=3e-3,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10
        )
