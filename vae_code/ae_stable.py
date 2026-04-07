'''
Importing Packages
'''
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, TensorDataset, Dataset
import xarray as xr
import pandas as pd
import math
import os
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

'''
Implementing the autoencoder
'''

# Helper classes and functions that are referred to within the vae function

class PairedAnomalyDataset(Dataset):
    def __init__(self, sst_tensor, precip_tensor):
        assert sst_tensor.shape[0] == precip_tensor.shape[0], "Time dimension mismatch"
        self.sst_tensor = sst_tensor
        self.precip_tensor = precip_tensor

    def __len__(self):
        return self.sst_tensor.shape[0]

    def __getitem__(self, idx):
        return self.sst_tensor[idx], self.precip_tensor[idx]


class MIMONet(nn.Module):
    def __init__(self, sst_dim, precip_dim):
        super(MIMONet, self).__init__()

        self.encoder_sst = nn.Sequential(
            nn.Linear(sst_dim, 50), nn.Tanh(), nn.Linear(50, 10), nn.Tanh()
        )
        self.encoder_precip = nn.Sequential(
            nn.Linear(precip_dim, 50), nn.Tanh(), nn.Linear(50, 10), nn.Tanh()
        )

        self.shared_latent = nn.Linear(20, 1)

        self.decoder_sst = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 50), nn.Tanh(), nn.Linear(50, sst_dim)
        )
        self.decoder_precip = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 50), nn.Tanh(), nn.Linear(50, precip_dim)
        )

    def forward(self, sst_x, precip_x):
        sst_encoded = self.encoder_sst(sst_x)
        precip_encoded = self.encoder_precip(precip_x)
        shared = self.shared_latent(torch.cat([sst_encoded, precip_encoded], dim=1))
        return self.decoder_sst(shared), self.decoder_precip(shared), shared
        
def plot_loss_curve(loss1, loss2, label1, label2, ylabel1, ylabel2, title, ax):
    ax.plot(loss1, label=label1, color='tab:blue', linewidth=0.75)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel1, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax.twinx()
    ax2.plot(loss2, label=label2, color='tab:orange', linewidth=0.75)
    ax2.set_ylabel(ylabel2, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax.set_title(title)
    
def plot_nd_loss_curve(loss_total, loss_sst, loss_precip, title, ax):
    ax.plot(loss_total, label="Total ND Loss", color='tab:gray', linewidth=0.75)
    ax.plot(loss_sst, label="SST ND Loss", color='tab:orange', linewidth=0.75)
    ax.plot(loss_precip, label="Precip ND Loss", color='tab:blue', linewidth=0.75)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Nondimensional Loss")
    ax.set_title(title)
    ax.legend()

def plot_latent(latent, title='Shared Latent Index from MIMO-AE', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))
        show_plot = True
    else:
        show_plot = False

    ax.plot(np.arange(latent.shape[0]), latent, label='MIMO-AE Index', linewidth=1, color='tab:gray')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Latent Value')
    ax.set_title(title)
    ax.legend()

    if show_plot:
        plt.tight_layout()
        plt.show()
        
def inverse_transform(scaler, tensor):
    return scaler.inverse_transform(tensor.detach().cpu().numpy())

'''
Defining the Autoencoder Itself
'''

def learn(sst_dat, precip_dat, norm, file_name=None, sst_var='sst', precip_var ='tp', train_pct=0.8, batch=32, epochs=100, verbose=True):
    import matplotlib.backends.backend_pdf

    # PREPARE AND TRANSFORM THE DATA FOR INTEGRATION INTO THE MODEL
    sst_np = np.nan_to_num(sst_dat[sst_var].values)
    precip_np = np.nan_to_num(precip_dat[precip_var].values)
    sst_flat = sst_np.reshape(sst_np.shape[0], -1)
    precip_flat = precip_np.reshape(precip_np.shape[0], -1)

    if norm == 'standard':
        sst_scaler, precip_scaler = StandardScaler(), StandardScaler()
    elif norm == 'minmax':
        sst_scaler, precip_scaler = MinMaxScaler(), MinMaxScaler()
    else:
        raise ValueError("norm must be 'standard' or 'minmax'")

    sst_scaled = sst_scaler.fit_transform(sst_flat)
    precip_scaled = precip_scaler.fit_transform(precip_flat)

    sst_tensor = torch.tensor(sst_scaled, dtype=torch.float32)
    precip_tensor = torch.tensor(precip_scaled, dtype=torch.float32)

    total_len = sst_tensor.shape[0]
    train_len = int(total_len * train_pct)
    sst_train, sst_test = sst_tensor[:train_len], sst_tensor[train_len:]
    precip_train, precip_test = precip_tensor[:train_len], precip_tensor[train_len:]

    train_loader = DataLoader(PairedAnomalyDataset(sst_train, precip_train), batch_size=batch, shuffle=False)
    test_loader = DataLoader(PairedAnomalyDataset(sst_test, precip_test), batch_size=batch, shuffle=False)

    model = MIMONet(sst_flat.shape[1], precip_flat.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    losses, precip_losses, sst_losses, nd_precip_losses, nd_sst_losses = [], [], [], [], []

    for epoch in range(epochs):
        epoch_total_loss, epoch_precip_loss, epoch_sst_loss, epoch_nd_p_loss, epoch_nd_s_loss, batch_count = 0, 0, 0, 0, 0, 0

        for sst_batch, precip_batch in train_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)
            sst_recon, precip_recon, _ = model(sst_batch, precip_batch)
            nd_s_loss, nd_p_loss = loss_fn(sst_recon, sst_batch), loss_fn(precip_recon, precip_batch)
            loss = nd_s_loss + nd_p_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Calculate MSE in physical units
            sst_loss = np.mean((inverse_transform(sst_scaler, sst_recon) - inverse_transform(sst_scaler, sst_batch)) ** 2)
            precip_loss = np.mean((inverse_transform(precip_scaler, precip_recon) - inverse_transform(precip_scaler, precip_batch)) ** 2)

            losses.append(loss.item())
            nd_sst_losses.append(nd_s_loss.item())
            nd_precip_losses.append(nd_p_loss.item())
            sst_losses.append(sst_loss)
            precip_losses.append(precip_loss)

            epoch_total_loss += loss.item()
            epoch_nd_s_loss += nd_s_loss.item()
            epoch_nd_p_loss += nd_p_loss.item()
            epoch_sst_loss += sst_loss
            epoch_precip_loss += precip_loss
            batch_count += 1

    model.eval()
    with torch.no_grad():
        latent_series = model.shared_latent(torch.cat([
            model.encoder_sst(sst_test.to(device)),
            model.encoder_precip(precip_test.to(device))
        ], dim=1)).cpu().numpy()

    test_losses, test_sst_losses, test_precip_losses = [], [], []
    with torch.no_grad():
        for sst_batch, precip_batch in test_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)
            sst_recon, precip_recon, _ = model(sst_batch, precip_batch)

            sst_loss = loss_fn(
                torch.tensor(inverse_transform(sst_scaler, sst_recon)).to(device),
                torch.tensor(inverse_transform(sst_scaler, sst_batch)).to(device)).item()
            precip_loss = loss_fn(
                torch.tensor(inverse_transform(precip_scaler, precip_recon)).to(device),
                torch.tensor(inverse_transform(precip_scaler, precip_batch)).to(device)).item()

            test_losses.append(sst_loss + precip_loss)
            test_sst_losses.append(sst_loss)
            test_precip_losses.append(precip_loss)

    # --- Save all visualizations to PDF ---
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"../signal-extraction-data/Results/{file_name}.pdf")
    fig, axes = plt.subplots(4, 1, figsize=(11, 14))
    
    # 1. Text summary
    axes[0].axis("off")
    text = (f"SST Loss: mean = {np.mean(test_sst_losses):.10f}, std = {np.std(test_sst_losses):.10f}\n"
            f"Precip Loss: mean = {np.mean(test_precip_losses):.10f}, std = {np.std(test_precip_losses):.10f}")
    axes[0].text(0.01, 0.6, text, fontsize=12)
    axes[0].set_title("Testing Loss Statistics")
    
    # 2. Training loss
    plot_loss_curve(
        loss1=precip_losses,
        loss2=sst_losses,
        label1="Precip Loss",
        label2="SST Loss",
        ylabel1="Precip Loss (m\u00B2)",
        ylabel2="SST Loss (\u00B0C\u00B2)",
        title="Training Loss of MIMO Autoencoder",
        ax=axes[1]
    )
    
    # 3. ND loss
    plot_nd_loss_curve(losses, nd_sst_losses, nd_precip_losses, "Training Nondimensional Losses", ax=axes[2])
    
    # 4. Latent plot
    plot_latent(latent_series, "Shared Latent Index from MIMO-AE", ax=axes[3])
    
    if file_name:
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        pdf.close()
   
    return {
        "model": model,
        "sst_scaler": sst_scaler,
        "precip_scaler": precip_scaler,
        "losses": losses,
        "nd_sst_losses": nd_sst_losses,
        "nd_precip_losses": nd_precip_losses,
        "sst_losses": sst_losses,
        "precip_losses": precip_losses,
        "latent_series": latent_series,
        "train_len": train_len
    }

'''
Latent Space extraction function
'''
def extractr(model, sst_dat, svar, precip_dat, pvar, sst_scaler, precip_scaler, plot=True, save=False, save_path=None):
    import xarray as xr
    import numpy as np
    import torch

    # Prepare inputs
    sst_np = np.nan_to_num(sst_dat[svar].values)
    precip_np = np.nan_to_num(precip_dat[pvar].values)

    sst_flat = sst_np.reshape(sst_np.shape[0], -1)
    precip_flat = precip_np.reshape(precip_np.shape[0], -1)

    sst_scaled = sst_scaler.transform(sst_flat)
    precip_scaled = precip_scaler.transform(precip_flat)

    sst_tensor = torch.tensor(sst_scaled, dtype=torch.float32)
    precip_tensor = torch.tensor(precip_scaled, dtype=torch.float32)

    # Send to device
    device = next(model.parameters()).device
    sst_tensor = sst_tensor.to(device)
    precip_tensor = precip_tensor.to(device)

    # Extract latent and reconstructions
    model.eval()
    with torch.no_grad():
        sst_recon, precip_recon, latent = model(sst_tensor, precip_tensor)

        sst_recon = sst_recon.cpu().numpy()
        precip_recon = precip_recon.cpu().numpy()
        latent = latent.cpu().numpy()

    # Inverse transform
    sst_recon = sst_scaler.inverse_transform(sst_recon).reshape(sst_np.shape)
    precip_recon = precip_scaler.inverse_transform(precip_recon).reshape(precip_np.shape)

    # Get time coordinate
    time = None
    for tname in ['time', 'valid_time']:
        if tname in sst_dat.coords:
            time = sst_dat[tname].values
            break
    if time is None:
        time = np.arange(latent.shape[0])

    # === Build xarray Datasets ===
    # Use original coordinates (time, lat, lon)
    coords_sst = dict(time=time, latitude=sst_dat.latitude.values, longitude=sst_dat.longitude.values)
    coords_precip = dict(time=time, latitude=precip_dat.latitude.values, longitude=precip_dat.longitude.values)

    dims = ('time', 'latitude', 'longitude')

    # SST Dataset
    sst_ds = xr.Dataset({
        'sst': (dims, sst_recon)
    }, coords=coords_sst)

    # Precip Dataset
    precip_ds = xr.Dataset({
        'precip': (dims, precip_recon)
    }, coords=coords_precip)

    # Latent time series
    latent_series = xr.DataArray(latent.squeeze(), coords={"time": time}, dims=["time"], name="latent")

    if plot:
        plot_latent(latent_series, title="Latent Series from New Dataset")

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save=True")
        latent_series.to_dataframe().reset_index().to_csv(f"{save_path}_latent.csv", index=False)
        sst_ds.to_netcdf(f"{save_path}_sst.nc")
        precip_ds.to_netcdf(f"{save_path}_precip.nc")

    return {
        "latent": latent_series,
        "sst": sst_ds,
        "precip": precip_ds
    }