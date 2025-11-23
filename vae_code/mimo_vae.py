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
Defining Helper Functions
'''
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

def plot_mu_logvar_z(mu_series, std_series, z_series, title_prefix="MIMOVAE"):
    """
    Plot latent mean μ(t), log variance logσ²(t), and latent sample z(t)
    with μ ± σ shading. Fully self-contained: creates its own figure/axes.
    """
    T = len(mu_series)
    t = np.arange(T)
    sigma_series = np.exp(0.5 * std_series)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # μ(t)
    axes[0].plot(t, mu_series, label="μ")
    axes[0].set_ylabel("μ")
    axes[0].set_title(f"{title_prefix}: Latent Mean (μ)")

    # logσ²(t)
    axes[1].plot(t, std_series, label="σ")
    axes[1].set_ylabel("σ")
    axes[1].set_title(f"{title_prefix}: Latent Standard Deviation (σ)")

    # z(t) + uncertainty band
    axes[2].plot(t, z_series, label="z", color="tab:green")
    axes[2].fill_between(
        t,
        mu_series.squeeze() - sigma_series.squeeze(),
        mu_series.squeeze() + sigma_series.squeeze(),
        color="gray", alpha=0.3, label="μ ± σ"
    )
    axes[2].set_ylabel("z")
    axes[2].set_xlabel("Time Index")
    axes[2].set_title(f"{title_prefix}: Latent Sample (z)")
    axes[2].legend()

    plt.tight_layout()
    return fig

def plot_training_losses(precip_losses, sst_losses, ax):
    plot_loss_curve(
        loss1=precip_losses,
        loss2=sst_losses,
        label1="Precip Loss",
        label2="SST Loss",
        ylabel1="Precip Loss (m^2)",
        ylabel2="SST Loss (ºC^2)",
        title="Training Loss of MIMO-VAE",
        ax=ax
    )

def plot_nd_losses(losses, nd_sst_losses, nd_precip_losses, ax):
    plot_nd_loss_curve(
        losses,
        nd_sst_losses,
        nd_precip_losses,
        "Training Nondimensional Losses",
        ax=ax
    )

def plot_summary(test_sst_losses, test_precip_losses, ax):
    ax.axis("off")
    text = (
        f"SST Loss: mean = {np.mean(test_sst_losses):.10f}, "
        f"std = {np.std(test_sst_losses):.10f}\n"
        f"Precip Loss: mean = {np.mean(test_precip_losses):.10f}, "
        f"std = {np.std(test_precip_losses):.10f}"
    )
    ax.text(0.01, 0.6, text, fontsize=12)
    ax.set_title("Testing Loss Statistics")

def train_summary(results, file_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        f"{file_name}.pdf"
    )
    
    # UNPACK RESULTS
    latent = results["z_series"]
    mu = results["mu_series"]
    std = np.exp(0.5 * results["logvar_series"])

    # FIGURE 1: SUMMARY + LOSSES 
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    # 1. summary statistics
    plot_summary(results["test_sst_losses"], results["test_precip_losses"], axes[0])

    # 2. physical training losses
    plot_training_losses(results["precip_losses"], results["sst_losses"], axes[1])

    # 3. nondimensional losses
    plot_nd_losses(results["losses"], results["nd_sst_losses"],
                   results["nd_precip_losses"], axes[2])

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # FIGURE 2: LATENT DIAGNOSTICS (μ, logσ², z with μ±σ)
    fig2 = plot_mu_logvar_z(mu, std, latent, title_prefix="MIMO-VAE")
    pdf.savefig(fig2)
    plt.close(fig2)

    pdf.close()

def inverse_transform(scaler, tensor):
    return scaler.inverse_transform(tensor.detach().cpu().numpy())

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


class MIMOVAE(nn.Module):
    def __init__(self, sst_dim, precip_dim):
        super(MIMOVAE, self).__init__()

        # --- Encoders stay the same ---
        self.encoder_sst = nn.Sequential(
            nn.Linear(sst_dim, 50), nn.Tanh(),
            nn.Linear(50, 10), nn.Tanh()
        )

        self.encoder_precip = nn.Sequential(
            nn.Linear(precip_dim, 50), nn.Tanh(),
            nn.Linear(50, 10), nn.Tanh()
        )

        # --- Shared hidden representation ---
        # (20 -> hidden_dim)
        self.shared_hidden = nn.Linear(20, 10)

        # --- Variational heads: μ and logσ² ---
        self.mu = nn.Linear(10, 1)         # 1-D latent mean
        self.logvar = nn.Linear(10, 1)     # 1-D latent log variance

        # --- Decoders stay the same ---
        self.decoder_sst = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(),
            nn.Linear(10, 50), nn.Tanh(),
            nn.Linear(50, sst_dim)
        )

        self.decoder_precip = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(),
            nn.Linear(10, 50), nn.Tanh(),
            nn.Linear(50, precip_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, sst_x, precip_x):
        # Encode each input
        sst_encoded = self.encoder_sst(sst_x)
        precip_encoded = self.encoder_precip(precip_x)

        # Shared hidden representation
        h = self.shared_hidden(torch.cat([sst_encoded, precip_encoded], dim=1))

        # μ and logσ²
        mu = self.mu(h)
        logvar = self.logvar(h)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        # Decode both outputs
        sst_hat = self.decoder_sst(z)
        precip_hat = self.decoder_precip(z)

        # Return decodings AND variational parameters
        return sst_hat, precip_hat, mu, logvar, z
        
'''
Defining the training function
'''
def learn(sst_dat, precip_dat, norm, sst_var='sst', precip_var='tp',
          train_pct=0.8, batch=32, epochs=100, verbose=True):

    # -------------------------------
    # DATA PREPARATION
    # -------------------------------
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

    train_loader = DataLoader(
        PairedAnomalyDataset(sst_train, precip_train),
        batch_size=batch,
        shuffle=False
    )
    test_loader = DataLoader(
        PairedAnomalyDataset(sst_test, precip_test),
        batch_size=batch,
        shuffle=False
    )

    # -------------------------------
    # MODEL + OPTIMIZER
    # -------------------------------
    model = MIMOVAE(sst_flat.shape[1], precip_flat.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # tracking
    losses, precip_losses, sst_losses = [], [], []
    nd_precip_losses, nd_sst_losses = [], []

    # -------------------------------
    # KL TERM
    # -------------------------------
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for epoch in range(epochs):
        for sst_batch, precip_batch in train_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)

            sst_recon, precip_recon, mu, logvar, z = model(sst_batch, precip_batch)

            nd_s_loss = loss_fn(sst_recon, sst_batch)
            nd_p_loss = loss_fn(precip_recon, precip_batch)
            kl = kl_loss(mu, logvar)

            loss = nd_s_loss + nd_p_loss + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # physical losses
            sst_loss = np.mean(
                (inverse_transform(sst_scaler, sst_recon) -
                 inverse_transform(sst_scaler, sst_batch)) ** 2
            )
            precip_loss = np.mean(
                (inverse_transform(precip_scaler, precip_recon) -
                 inverse_transform(precip_scaler, precip_batch)) ** 2
            )

            losses.append(loss.item())
            nd_sst_losses.append(nd_s_loss.item())
            nd_precip_losses.append(nd_p_loss.item())
            sst_losses.append(sst_loss)
            precip_losses.append(precip_loss)

    # -------------------------------
    # LATENT EXTRACTION (mu, logvar, z)
    # -------------------------------
    model.eval()
    mu_list, logvar_list, z_list = [], [], []

    with torch.no_grad():
        for sst_batch, precip_batch in test_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)
            _, _, mu, logvar, z = model(sst_batch, precip_batch)
            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())
            z_list.append(z.cpu().numpy())

    mu_series = np.concatenate(mu_list, axis=0)
    logvar_series = np.concatenate(logvar_list, axis=0)
    z_series = np.concatenate(z_list, axis=0)

    # -------------------------------
    # TEST LOSSES
    # -------------------------------
    test_sst_losses, test_precip_losses = [], []

    with torch.no_grad():
        for sst_batch, precip_batch in test_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)
            sst_recon, precip_recon, mu, logvar, z = model(sst_batch, precip_batch)

            sst_loss = loss_fn(
                torch.tensor(inverse_transform(sst_scaler, sst_recon)).to(device),
                torch.tensor(inverse_transform(sst_scaler, sst_batch)).to(device)
            ).item()

            precip_loss = loss_fn(
                torch.tensor(inverse_transform(precip_scaler, precip_recon)).to(device),
                torch.tensor(inverse_transform(precip_scaler, precip_batch)).to(device)
            ).item()

            test_sst_losses.append(sst_loss)
            test_precip_losses.append(precip_loss)

    # -------------------------------
    # RETURN EVERYTHING NEEDED FOR PLOTTING
    # -------------------------------
    return {
        "model": model,
        "sst_scaler": sst_scaler,
        "precip_scaler": precip_scaler,
        "losses": losses,
        "nd_sst_losses": nd_sst_losses,
        "nd_precip_losses": nd_precip_losses,
        "sst_losses": sst_losses,
        "precip_losses": precip_losses,
        "test_sst_losses": test_sst_losses,
        "test_precip_losses": test_precip_losses,
        "mu_series": mu_series,
        "logvar_series": logvar_series,
        "z_series": z_series,
        "train_len": train_len
    }

def extractr(model, sst_dat, svar, precip_dat, pvar, sst_scaler, precip_scaler,
             plot=True, save=False, save_path=None):
    import xarray as xr
    import numpy as np
    import torch

    # -------------------------------
    # PREPARE INPUTS
    # -------------------------------
    sst_np = np.nan_to_num(sst_dat[svar].values)
    precip_np = np.nan_to_num(precip_dat[pvar].values)

    sst_flat = sst_np.reshape(sst_np.shape[0], -1)
    precip_flat = precip_np.reshape(precip_np.shape[0], -1)

    sst_scaled = sst_scaler.transform(sst_flat)
    precip_scaled = precip_scaler.transform(precip_flat)

    sst_tensor = torch.tensor(sst_scaled, dtype=torch.float32)
    precip_tensor = torch.tensor(precip_scaled, dtype=torch.float32)

    device = next(model.parameters()).device
    sst_tensor = sst_tensor.to(device)
    precip_tensor = precip_tensor.to(device)

    # -------------------------------
    # FORWARD PASS THROUGH MIMOVAE
    # -------------------------------
    model.eval()
    with torch.no_grad():
        sst_recon, precip_recon, mu, logvar, z = model(sst_tensor, precip_tensor)

        sst_recon = sst_recon.cpu().numpy()
        precip_recon = precip_recon.cpu().numpy()
        mu_series = mu.cpu().numpy().squeeze()
        logvar_series = logvar.cpu().numpy().squeeze()
        z_series = z.cpu().numpy().squeeze()

    # -------------------------------
    # INVERSE TRANSFORM RECONSTRUCTIONS
    # -------------------------------
    sst_recon = sst_scaler.inverse_transform(sst_recon).reshape(sst_np.shape)
    precip_recon = precip_scaler.inverse_transform(precip_recon).reshape(precip_np.shape)

    # -------------------------------
    # TIME COORDINATE
    # -------------------------------
    time = None
    for key in ["time", "valid_time"]:
        if key in sst_dat.coords:
            time = sst_dat[key].values
            break
    if time is None:
        time = np.arange(len(z_series))

    # -------------------------------
    # BUILD XARRAY RECONSTRUCTED FIELDS
    # -------------------------------
    def get_coord(ds, *names):
        for name in names:
            if name in ds.coords:
                return ds[name].values
        raise KeyError(names)

    lat_sst  = get_coord(sst_dat,  "lat", "latitude")
    lon_sst  = get_coord(sst_dat,  "lon", "longitude")
    lat_prec = get_coord(precip_dat, "lat", "latitude")
    lon_prec = get_coord(precip_dat, "lon", "longitude")
    
    coords_sst    = dict(time=time, latitude=lat_sst,  longitude=lon_sst)
    coords_precip = dict(time=time, latitude=lat_prec, longitude=lon_prec)
    
    dims = ("time", "latitude", "longitude")

    sst_ds = xr.Dataset({"sst": (dims, sst_recon)}, coords=coords_sst)
    precip_ds = xr.Dataset({"precip": (dims, precip_recon)}, coords=coords_precip)

    # -------------------------------
    # UNIFIED LATENT TABLE (4 COLUMNS)
    # -------------------------------
    latent_ds = xr.Dataset(
        {
            "latent": ("time", z_series),
            "mu": ("time", mu_series),
            "logvar": ("time", logvar_series),
        },
        coords={"time": time}
    )

    # -------------------------------
    # OPTIONAL PLOTTING
    # -------------------------------
    if plot:
        plot_mu_logvar_z(latent_ds["mu"].values, latent_ds["logvar"].values, latent_ds["latent"].values, title_prefix="MIMOVAE")
        
    # -------------------------------
    # OPTIONAL SAVE TO A SINGLE CSV
    # -------------------------------
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save=True")

        # save latent table
        latent_df = latent_ds.to_dataframe().reset_index()
        latent_df.to_csv(f"{save_path}_latent_full.csv", index=False)

        # save fields
        sst_ds.to_netcdf(f"{save_path}_sst.nc")
        precip_ds.to_netcdf(f"{save_path}_precip.nc")

    # -------------------------------
    # RETURN EVERYTHING
    # -------------------------------
    return {
        "latent": latent_ds,     # unified 4-column table
        "sst": sst_ds,
        "precip": precip_ds
    }
