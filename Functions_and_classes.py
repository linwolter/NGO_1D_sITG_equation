import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from torch import nn, optim, utils
from torch.utils.data import Dataset, DataLoader, random_split
import opt_einsum


# Dataset loading function that provides the full time in one part
class PDEDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        params = data['params']
        initial_conditions = data['initial_conditions']    
        solutions = data['solutions'] 
        
        self.x = torch.from_numpy(params).float()
        
        # Handle complex-valued solutions
        if np.iscomplexobj(solutions):
            if solutions.ndim == 3:
                solutions = solutions.reshape(solutions.shape[0], -1)
            
            self.y = torch.complex(
                torch.from_numpy(solutions.real).float(),
                torch.from_numpy(solutions.imag).float()
            )
        else: #back-up for some weird case where data would not be complex
            self.y = torch.from_numpy(solutions).float()
            if self.y.ndim == 5:
                self.y = self.y.view(self.y.size(0), -1)

        if np.iscomplexobj(initial_conditions):
            if initial_conditions.ndim == 3:
                initial_conditions = initial_conditions.reshape(initial_conditions.shape[0], -1)

            self.z = torch.complex(
                torch.from_numpy(initial_conditions.real).float(),
                torch.from_numpy(initial_conditions.imag).float()
            )
        
        assert self.x.shape[0] == self.y.shape[0] ==self.z.shape[0], \
            f"Mismatch: inputs {self.x.shape[0]} vs outputs {self.y.shape[0]}"
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.z[idx], self.y[idx]

# Basic integrate function taken from Joost Prins
def integrate(W, values):
    """
    Values shape: (Batch, Time, Velocity) -> (B, 3, 32)
    W shape: (Velocity,) -> (32,)
    Returns: (Batch, Time)
    """
    # Simple trapezoidal-style integration or weighted sum
    # We sum over the last dimension (velocity)
    return torch.sum(values * W, dim=-1)

# Basic L2 loss function also taken from Joost Prins
def relative_l2_loss(y_pred, y_true):
# y shape: (batch, time, velocity)
# Calculate error and norm over the velocity/time dimensions
    diff_norm = torch.norm(y_pred - y_true, p=2, dim=(1, 2))
    true_norm = torch.norm(y_true, p=2, dim=(1, 2))
    return torch.mean(diff_norm / (true_norm + 1e-8))

# Dataset creation function taking data file as input, now cuts the full time into parts
class PDEDataset_timestep(Dataset):
    def __init__(self, file_path, Nt_block=2, mode='train'):
        """
        Nt_block: The number of time steps the model will predict in one forward pass.
        mode: 'train' enables random time-slicing. 'val'/'test' starts from t=0.
        """
        super().__init__()
        data = np.load(file_path)
        params = data['params']
        solutions = data['solutions'] 
        
        self.Nt_block = Nt_block
        self.mode = mode
        self.x = torch.from_numpy(params).float()
        
        # Complex numbers dealt with by separating
        if np.iscomplexobj(solutions):
            self.y = torch.complex(
                torch.from_numpy(solutions.real).float(),
                torch.from_numpy(solutions.imag).float()
            )
        else:
            self.y = torch.from_numpy(solutions).float()

        # make sure that the shapes match
        assert self.x.shape[0] == self.y.shape[0], \
            f"Mismatch: inputs {self.x.shape[0]} vs outputs {self.y.shape[0]}"
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        param = self.x[idx]
        full_sol = self.y[idx]  # Shape: (Nt, Nv)
        
        total_time_steps = full_sol.shape[0]
        
        if self.mode == 'train':
            # Pick a random starting index, leaving enough room for the block
            max_start_idx = total_time_steps - self.Nt_block
            start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
        else:
            # For validation/testing, we usually want to evaluate starting from t=0
            start_idx = 0
            
        # The target is the slice of the solution for this time block
        target_block = full_sol[start_idx : start_idx + self.Nt_block, :] # (Nt_block, Nv)
        
        # The initial condition for this block is just the first time step of the slice
        ic = target_block[0, :] # (Nv,)
        
        return param, ic, target_block
    


