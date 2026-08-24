#imports
from model_setup import  VlasovModel_Compact,VlasovModel_CNN,VlasovModel_MLP_simple
from Functions_and_classes import PDEDataset, DataLoader, PDEDataset_timestep
data_path = "vlasov_ngo_data_64x64.npz"

import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from torch import nn, optim, utils
from torch.utils.data import Dataset, DataLoader, random_split
import opt_einsum

from basis_functions_classes import make_basis_matrix, make_basis_matrix_piecewise_linear, make_hermite_legendre_basis


#parameters
Nt, Nv = 8,64
N_basis_t = 8 
N_basis_v = 64
time = 1.25

basis_matrix,basis_t,basis_v = make_basis_matrix_piecewise_linear(Nt=Nt,Nv=Nv,N_basis_t=N_basis_t,N_basis_v=N_basis_v,time=time)
#basis_matrix = make_basis_matrix(Nt,Nv,N_basis_t,N_basis_v)

from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # Either timestep data or full data
    full_dataset = PDEDataset_timestep(data_path, Nt_block=8)
    #full_dataset = PDEDataset(data_path)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    # checkpoint callback, save best val_loss automatically
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="vlasov_compact_best",
        save_top_k=1,        # only keep the single best
        mode="min",
    )

    # Choose which model
    model = VlasovModel_CNN(
        input_dim=4, Nv=Nv, Nt=Nt,
        basis_matrix=basis_matrix
    )

    #model = VlasovModel_MLP_simple(input_dim=4,Nv=Nv,Nt=64)

    # model = VlasovModel_Compact(
    #     input_dim=4, Nv=Nv, Nt=Nt, N_basis_t=N_basis_t, N_basis_v=N_basis_v,
    #     basis_matrix=basis_matrix, rank=64
    # )

    # Set the training
    trainer = L.Trainer(
        max_epochs=3000,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    #Just define which model is being saved with lowest loss
    best_model = VlasovModel_CNN.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        input_dim=4,
        Nv=Nv,
        Nt=Nt,
        basis_matrix=basis_matrix
    )

    #best_model = VlasovModel_MLP_simple.load_from_checkpoint(checkpoint_callback.best_model_path,input_dim=4, Nv=Nv, Nt=64)

    # best_model = VlasovModel_Compact.load_from_checkpoint(
    #     checkpoint_callback.best_model_path,input_dim=4, Nv=Nv, Nt=Nt, N_basis_t=N_basis_t, N_basis_v=N_basis_v,
    #     basis_matrix=basis_matrix, rank=64
    # )

    # Switch to eval mode to freeze dropout/batchnorm
    best_model.eval()


    # Set name to save model
    torch.save(best_model.state_dict(), "vlasov_model_CNN_3000_0707_0exploss.pth")