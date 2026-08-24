import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from torch import nn, optim, utils
from torch.utils.data import Dataset, DataLoader, random_split
import opt_einsum
import torch.nn.functional as F
from pytorch_optimizer import SOAP

from Functions_and_classes import integrate, relative_l2_loss, block_relative_l2_loss, log_l2_loss, complex_asinh_mse, residual_aware_loss

def time_weighted_relative_loss(y_pred, y_true):

    # Norm at each timestep independently
    num = torch.norm(y_pred - y_true, p=2, dim=-1)  
    den = torch.norm(y_true,          p=2, dim=-1).clamp(min=1e-8) 
    
    per_t_loss = num / den        
    return per_t_loss.mean()
    
# A class called compact due to the approximated nature of the Greens matrix
class VlasovModel_Compact(L.LightningModule):
    def __init__(self, input_dim=4, Nv=32, Nt=16, N_basis_t=16, N_basis_v=32,
                 basis_matrix=None, rank=8):
        super().__init__()
        
        self.K = N_basis_t * N_basis_v
        self.rank = rank             
        self.Nt = Nt
        self.Nv = Nv
        self.N_basis_t = N_basis_t
        self.N_basis_v = N_basis_v
        
        self.register_buffer('basis_real', basis_matrix)

        # MLP to encode params
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),        nn.ReLU(),
        )  

        # Diagonal head: predict 2K values (real + imag)
        self.head_diag = nn.Linear(64, self.K * 2)

        # Low-rank heads: predict separable t and v components
        # U_t: (N_basis_t × rank), U_v: (N_basis_v × rank)
        # Final U[:, j] = kron(U_t[:, j], U_v[:, j])  →  shape (K, rank)
        self.head_Ut_re = nn.Linear(64, N_basis_t * rank)
        self.head_Ut_im = nn.Linear(64, N_basis_t * rank)
        self.head_Uv_re = nn.Linear(64, N_basis_v * rank)
        self.head_Uv_im = nn.Linear(64, N_basis_v * rank)

        self.head_Vt_re = nn.Linear(64, N_basis_t * rank)
        self.head_Vt_im = nn.Linear(64, N_basis_t * rank)
        self.head_Vv_re = nn.Linear(64, N_basis_v * rank)
        self.head_Vv_im = nn.Linear(64, N_basis_v * rank)

    def _build_separable_UV(self, z, batch_size):
        """
        Build U, V as Kronecker-separable matrices.
        U[:, j] = kron(U_t[:, j], U_v[:, j])
        Shape: (batch, K, rank), complex
        """
        r = self.rank
        Nt, Nv = self.N_basis_t, self.N_basis_v

        # Decode t and v factors
        Ut_re = self.head_Ut_re(z).view(batch_size, Nt, r)
        Ut_im = self.head_Ut_im(z).view(batch_size, Nt, r)
        Uv_re = self.head_Uv_re(z).view(batch_size, Nv, r)
        Uv_im = self.head_Uv_im(z).view(batch_size, Nv, r)

        Vt_re = self.head_Vt_re(z).view(batch_size, Nt, r)
        Vt_im = self.head_Vt_im(z).view(batch_size, Nt, r)
        Vv_re = self.head_Vv_re(z).view(batch_size, Nv, r)
        Vv_im = self.head_Vv_im(z).view(batch_size, Nv, r)

        # Complex factors
        Ut = torch.complex(Ut_re, Ut_im)  
        Uv = torch.complex(Uv_re, Uv_im)  
        Vt = torch.complex(Vt_re, Vt_im)
        Vv = torch.complex(Vv_re, Vv_im)

        # Kronecker product per rank: (B, Nt, r) ⊗ (B, Nv, r) → (B, Nt*Nv, r)
        # kron along spatial dims: U[:, i*Nv + j, k] = Ut[:, i, k] * Uv[:, j, k]
        U = (Ut.unsqueeze(2) * Uv.unsqueeze(1)).view(batch_size, Nt * Nv, r) 
        V = (Vt.unsqueeze(2) * Vv.unsqueeze(1)).view(batch_size, Nt * Nv, r)

        return U, V  # both (B, K, rank), complex
    
    def compute_d_n(self, initial_conditions):
        """Construct right hand side vector d_n"""
        batch_size = initial_conditions.shape[0]
        v = torch.linspace(-4.0, 4.0, self.Nv, device=self.device)
        dv = v[1] - v[0]
        
        w_v = torch.ones(self.Nv, device=self.device) * dv
        w_v[0] *= 0.5
        w_v[-1] *= 0.5

        basis_at_t0 = self.basis_real.reshape(self.K, self.Nt, self.Nv)[:, 0, :]
        f0 = initial_conditions.view(batch_size, self.Nv)
        
        # Projection: d_n = B_t0 * W * f0
        d_n = (basis_at_t0 * w_v).to(f0.dtype) @ f0.T 
        return d_n.T 

    def forward(self, params, initial_conditions):
        batch_size = params.shape[0]
        d_n = self.compute_d_n(initial_conditions)  
        z = self.mlp(params) 

        # Diagonal
        diag_out = self.head_diag(z)
        d_re = diag_out[..., :self.K]
        d_im = diag_out[..., self.K:]
        d_cplx = torch.complex(d_re, d_im)  # (B, K)

        # Low-rank
        U, V = self._build_separable_UV(z, batch_size) 

        # Apply A = diag(d) + U @ Vᴴ  to d_n
        d_n_c = d_n.to(U.dtype)                                    
        diag_term = d_cplx * d_n_c                                  
        rank_term = (U @ (V.mH @ d_n_c.unsqueeze(-1))).squeeze(-1)   
        c = diag_term + rank_term                                      

        basis = self.basis_real.to(c.dtype)
        return c @ basis 

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

        return torch.cat(all_preds, dim=1)  # output now contains one less time step per block, using it for predictions require taking this into account

    def training_step(self, batch, batch_idx):
        params, initial_conditions, solution = batch
        if solution.dim() == 2:
            solution = solution.view(solution.shape[0], -1, self.Nv)
        
        num_blocks = solution.shape[1] // self.Nt
        y_hat = self.rollout(params, initial_conditions, num_blocks) 
        
        # simple rel l2 loss 
        loss_shape = time_weighted_relative_loss(y_hat, solution)
        
        # Loss on the growth rate
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

        # Rel l2 loss 
        loss_shape = time_weighted_relative_loss(y_hat, solution)
        
        # Loss on growth rate
        log_norm_pred = torch.log(torch.norm(y_hat,     p=2, dim=-1).clamp(min=1e-8))
        log_norm_true = torch.log(torch.norm(solution,  p=2, dim=-1).clamp(min=1e-8))
        loss_growth = F.mse_loss(log_norm_pred, log_norm_true)
        
        loss = loss_shape + 0.1 * loss_growth

        self.log("val_loss", loss, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     return optim.Adam([
    #         {'params': self.mlp.parameters(),       'weight_decay': 0,    'lr': 1e-3},
    #         {'params': self.head_diag.parameters(), 'weight_decay': 0,    'lr': 1e-3},
    #         {'params': [*self.head_Ut_re.parameters(), *self.head_Ut_im.parameters(),
    #                     *self.head_Uv_re.parameters(), *self.head_Uv_im.parameters(),
    #                     *self.head_Vt_re.parameters(), *self.head_Vt_im.parameters(),
    #                     *self.head_Vv_re.parameters(), *self.head_Vv_im.parameters()],
    #         'weight_decay': 1e-4, 'lr': 1e-3},
    #     ], lr=1e-3)

    def configure_optimizers(self):
        return SOAP([
            {
                'params': self.mlp.parameters(),       
                'weight_decay': 0,    
                'lr': 1e-3
            },
            {
                'params': self.head_diag.parameters(), 
                'weight_decay': 0,    
                'lr': 1e-3
            },
            {
                'params': [
                    *self.head_Ut_re.parameters(), *self.head_Ut_im.parameters(),
                    *self.head_Uv_re.parameters(), *self.head_Uv_im.parameters(),
                    *self.head_Vt_re.parameters(), *self.head_Vt_im.parameters(),
                    *self.head_Vv_re.parameters(), *self.head_Vv_im.parameters()
                ],
                'weight_decay': 1e-4, 
                'lr': 1e-3
            },
        ], 
        # Global defaults for the optimizer (applies to any params not explicitly grouped above)
        lr=3e-3, 
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class VlasovModel_CNN(L.LightningModule):
    def __init__(self, input_dim=4, Nv=32, Nt=16, basis_matrix=None):
        super().__init__()
        self.save_hyperparameters(ignore=['basis_matrix'])
        
        self.Nt = Nt
        self.Nv = Nv
        self.K = basis_matrix.shape[0] if basis_matrix is not None else 0
        self.register_buffer('basis_real', basis_matrix)


        self.branch_full = nn.Sequential(
            # 1. Project input (4) to bottleneck (256)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            Reshape((256, 1, 1)),

            # 2. Layer 4: Expand to 8x8
            nn.ConvTranspose2d(256, 64, kernel_size=8, stride=8, bias=False),
            nn.ReLU(),

            # 3. Layer 5: Expand to 32x32
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4, bias=False),
            nn.ReLU(),

            # 4. Layer 6: Expand to 128x128
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4, bias=False),
            nn.ReLU(),

            # 5. Layer 7: Expand to 512x512
            # Note: out_channels=2 gives us Real and Imaginary parts directly
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=4, bias=False)
        )


    def compute_d_n(self, initial_conditions):
        """Create right hand side vector d_n"""
        batch_size = initial_conditions.shape[0]
        v = torch.linspace(-4.0, 4.0, self.Nv, device=self.device)
        dv = v[1] - v[0]
        
        # Trapezoidal weights
        w_v = torch.ones(self.Nv, device=self.device) * dv
        w_v[0] *= 0.5
        w_v[-1] *= 0.5

        basis_at_t0 = self.basis_real.reshape(self.K, self.Nt, self.Nv)[:, 0, :]
        f0 = initial_conditions.view(batch_size, self.Nv)
        
        # d_n = B_t0 * W * f0
        d_n = (basis_at_t0 * w_v).to(f0.dtype) @ f0.T 
        return d_n.T 

    def _apply_full_matrix(self, params, d_n):
        batch_size = params.shape[0]
        
        # cnn_out shape: (Batch, 2, 512, 512)
        cnn_out = self.branch_full(params)
        
        # Split the channels into real and imaginary
        A_real = cnn_out[:, 0, :, :]
        A_imag = cnn_out[:, 1, :, :] 
        
        A = torch.complex(A_real, A_imag)
        
        # A * d_n
        return torch.bmm(A, d_n.to(A.dtype).unsqueeze(-1)).squeeze(-1)


    def forward(self, params, initial_conditions):
        d_n = self.compute_d_n(initial_conditions)
   
        c = self._apply_full_matrix(params, d_n)

        basis = self.basis_real.to(c.dtype)
        return c @ basis 

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

        return torch.cat(all_preds, dim=1)  # Out now contains one less time step per block
    
    
    def training_step(self, batch, batch_idx):
        params, initial_conditions, solution = batch
        if solution.dim() == 2:
            solution = solution.view(solution.shape[0], -1, self.Nv)
        
        num_blocks = solution.shape[1] // self.Nt
        y_hat = self.rollout(params, initial_conditions, num_blocks) 
        
        # L2 rel per time chunk
        loss_shape = time_weighted_relative_loss(y_hat, solution)
        
        # Growth rate loss
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
        
        # Rel l2 loss per time chunk
        loss_shape = time_weighted_relative_loss(y_hat, solution)
        
        # Growth rate loss
        log_norm_pred = torch.log(torch.norm(y_hat,     p=2, dim=-1).clamp(min=1e-8))
        log_norm_true = torch.log(torch.norm(solution,  p=2, dim=-1).clamp(min=1e-8))
        loss_growth = F.mse_loss(log_norm_pred, log_norm_true)
        
        loss = loss_shape + 0.1 * loss_growth
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)



        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return SOAP(
        self.parameters(),
        lr=3e-3,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10
        )


class VlasovModel_MLP_simple(L.LightningModule):
    def __init__(self, input_dim=4, Nv=32, Nt=64, hidden_dim=512, num_layers=3):
        super().__init__()
        self.save_hyperparameters()
        self.Nt = Nt
        self.Nv = Nv

        in_features = input_dim + 2 * Nv
        out_features = Nt * Nv

        def build_layers():
            layers = []
            for i in range(num_layers):
                layers.append(nn.Linear(in_features if i == 0 else hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_features))
            return nn.Sequential(*layers)

        self.net_real = build_layers()
        self.net_imag = build_layers()

    def forward(self, params, initial_conditions):
        # Feed all inputs and define what the output should be
        ic_real = initial_conditions.real
        ic_imag = initial_conditions.imag  
        x = torch.cat([params, ic_real, ic_imag], dim=-1)  # Make it a vector
        out = torch.complex(self.net_real(x), self.net_imag(x))  
        return out.view(-1, self.Nt, self.Nv)                    

    def _compute_loss(self, batch):
        params, initial_conditions, solution = batch
        if solution.dim() == 2:
            solution = solution.view(solution.shape[0], self.Nt, self.Nv)
        y_hat = self(params, initial_conditions)
        return time_weighted_relative_loss(y_hat, solution)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-3, weight_decay=0.01)