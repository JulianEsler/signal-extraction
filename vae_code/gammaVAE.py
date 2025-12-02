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

"""
Helper Functions (Gamma-latent compatible)
"""

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


def plot_alpha_beta_z(alpha_series, beta_series, z_series, title_prefix="MIMOVAE"):
    """
    Plot alpha(t), beta(t), and z(t).
    Derived Gaussian-style uncertainty bands are shown using:
      mean = alpha / beta
      std  = sqrt(alpha) / beta
    """
    T = len(z_series)
    t = np.arange(T)

    mu = alpha_series / beta_series
    std = np.sqrt(alpha_series) / beta_series

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Alpha (shape)
    axes[0].plot(t, alpha_series, label="alpha (shape)")
    axes[0].set_ylabel("alpha")
    axes[0].set_title(f"{title_prefix}: Latent Shape (alpha)")

    # Beta (rate)
    axes[1].plot(t, beta_series, label="beta (rate)")
    axes[1].set_ylabel("beta")
    axes[1].set_title(f"{title_prefix}: Latent Rate (beta)")

    # z(t) with uncertainty band mu ± std
    axes[2].plot(t, z_series, color="tab:green", label="z")
    axes[2].fill_between(
        t,
        (mu - std).squeeze(),
        (mu + std).squeeze(),
        alpha=0.3,
        color="gray",
        label="Gamma mean ± std"
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
        ylabel1="Precip Loss (physical)",
        ylabel2="SST Loss (physical)",
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
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"{file_name}.pdf")

    # -------------------------------------------------------
    # unpack (Gamma latent)
    # -------------------------------------------------------
    latent = results["z_series"]
    alpha = results["alpha_series"]
    beta = results["beta_series"]

    # FIGURE 1: summary + losses
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    plot_summary(results["test_sst_losses"], results["test_precip_losses"], axes[0])
    plot_training_losses(results["precip_losses"], results["sst_losses"], axes[1])
    plot_nd_losses(results["losses"], results["nd_sst_losses"], results["nd_precip_losses"], axes[2])

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # FIGURE 2: latent diagnostics for Gamma latent
    fig2 = plot_alpha_beta_z(alpha, beta, latent, title_prefix="MIMO-VAE (Gamma Latent)")
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

        # --- Encoders (unchanged) ---
        self.encoder_sst = nn.Sequential(
            nn.Linear(sst_dim, 50), nn.Tanh(),
            nn.Linear(50, 10), nn.Tanh()
        )

        self.encoder_precip = nn.Sequential(
            nn.Linear(precip_dim, 50), nn.Tanh(),
            nn.Linear(50, 10), nn.Tanh()
        )

        # --- Shared hidden representation ---
        self.shared_hidden = nn.Linear(20, 10)

        # --- Variational heads: alpha, beta for Gamma posterior ---
        # Use Softplus to ensure positivity
        self.alpha_head = nn.Sequential(
            nn.Linear(10, 1),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(10, 1),
            nn.Softplus()
        )

        # --- Decoders (unchanged) ---
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

    def reparameterize(self, alpha, beta):
        """
        Gamma posterior q(z|x) = Gamma(alpha, beta) with rate parameterization.
        Use PyTorch's rsample() (implicit reparameterization).
        """
        dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
        z = dist.rsample()  # differentiable w.r.t. alpha, beta
        return z

    def forward(self, sst_x, precip_x):
        # Encode each input
        sst_encoded = self.encoder_sst(sst_x)
        precip_encoded = self.encoder_precip(precip_x)

        # Shared hidden representation
        h = self.shared_hidden(torch.cat([sst_encoded, precip_encoded], dim=1))

        # Gamma parameters (posterior)
        alpha = self.alpha_head(h) + 1e-4  # avoid exact zeros
        beta = self.beta_head(h) + 1e-4

        # Latent sample
        z = self.reparameterize(alpha, beta)

        # Decode both outputs
        sst_hat = self.decoder_sst(z)
        precip_hat = self.decoder_precip(z)

        # Return decodings AND variational parameters
        return sst_hat, precip_hat, alpha, beta, z

        
'''
Defining the training function
'''
def learn(sst_dat, precip_dat, norm, sst_var='sst', precip_var='tp',
          train_pct=0.8, batch=32, epochs=100, verbose=True):

    # -------------------------------
    # DATA PREPARATION (unchanged)
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
    # KL TERM: Gamma posterior vs Gamma prior
    # -------------------------------
    # Prior parameters (can be tuned)
    prior_alpha = torch.tensor(1.0, device=device)  # shape
    prior_beta = torch.tensor(1.0, device=device)   # rate

    def gamma_kl(alpha_q, beta_q, alpha_p, beta_p):
        """
        KL( Gamma(alpha_q, beta_q) || Gamma(alpha_p, beta_p) )
        shape-rate parameterization.
        """
        # ensure broadcast
        alpha_p_b = alpha_p.expand_as(alpha_q)
        beta_p_b = beta_p.expand_as(beta_q)

        term1 = (alpha_q - alpha_p_b) * torch.digamma(alpha_q)
        term2 = torch.lgamma(alpha_p_b) - torch.lgamma(alpha_q)
        term3 = alpha_p_b * (torch.log(beta_q) - torch.log(beta_p_b))
        term4 = alpha_q * (beta_p_b / beta_q - 1.0)

        return term1 + term2 + term3 + term4  # shape: (batch, 1)

    def kl_loss(alpha_q, beta_q):
        kl_per_sample = gamma_kl(alpha_q, beta_q, prior_alpha, prior_beta)
        return kl_per_sample.mean()

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for epoch in range(epochs):
        for sst_batch, precip_batch in train_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)

            # Note: model now returns alpha, beta instead of mu, logvar
            sst_recon, precip_recon, alpha, beta, z = model(sst_batch, precip_batch)

            nd_s_loss = loss_fn(sst_recon, sst_batch)
            nd_p_loss = loss_fn(precip_recon, precip_batch)
            kl = kl_loss(alpha, beta)

            loss = nd_s_loss + nd_p_loss + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # physical losses (unchanged)
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
    # LATENT EXTRACTION (alpha, beta, z)
    # -------------------------------
    model.eval()
    alpha_list, beta_list, z_list = [], [], []

    with torch.no_grad():
        for sst_batch, precip_batch in test_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)
            _, _, alpha, beta, z = model(sst_batch, precip_batch)
            alpha_list.append(alpha.cpu().numpy())
            beta_list.append(beta.cpu().numpy())
            z_list.append(z.cpu().numpy())

    alpha_series = np.concatenate(alpha_list, axis=0)
    beta_series = np.concatenate(beta_list, axis=0)
    z_series = np.concatenate(z_list, axis=0)

    # -------------------------------
    # TEST LOSSES (unchanged)
    # -------------------------------
    test_sst_losses, test_precip_losses = [], []

    with torch.no_grad():
        for sst_batch, precip_batch in test_loader:
            sst_batch, precip_batch = sst_batch.to(device), precip_batch.to(device)
            sst_recon, precip_recon, alpha, beta, z = model(sst_batch, precip_batch)

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
    # RETURN EVERYTHING
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
        "alpha_series": alpha_series,
        "beta_series": beta_series,
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
    # (Gamma latent: returns alpha, beta instead of mu, logvar)
    # -------------------------------
    model.eval()
    with torch.no_grad():
        sst_recon, precip_recon, alpha, beta, z = model(sst_tensor, precip_tensor)

        sst_recon = sst_recon.cpu().numpy()
        precip_recon = precip_recon.cpu().numpy()
        alpha_series = alpha.cpu().numpy().squeeze()
        beta_series  = beta.cpu().numpy().squeeze()
        z_series     = z.cpu().numpy().squeeze()

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

    lat_sst  = get_coord(sst_dat, "lat", "latitude")
    lon_sst  = get_coord(sst_dat, "lon", "longitude")
    lat_prec = get_coord(precip_dat, "lat", "latitude")
    lon_prec = get_coord(precip_dat, "lon", "longitude")

    coords_sst    = dict(time=time, latitude=lat_sst, longitude=lon_sst)
    coords_precip = dict(time=time, latitude=lat_prec, longitude=lon_prec)
    dims = ("time", "latitude", "longitude")

    sst_ds = xr.Dataset({"sst": (dims, sst_recon)}, coords=coords_sst)
    precip_ds = xr.Dataset({"precip": (dims, precip_recon)}, coords=coords_precip)

    # -------------------------------
    # UNIFIED LATENT TABLE (Gamma latent)
    #
    # alpha  = shape parameter of q(z|x)
    # beta   = rate  parameter of q(z|x)
    #
    # lambda = posterior mean of the Gamma latent:
    #          E[z | x] = alpha / beta
    #          This is the recommended variable for conditioning
    #          external models (e.g., the flood simulator), because
    #          it removes sampling noise and reflects the VAE's
    #          learned latent intensity signal.
    # -------------------------------
    
    lambda_series = alpha_series / beta_series   # posterior mean
    
    latent_ds = xr.Dataset(
        {
            "latent":  ("time", z_series),
            "alpha":   ("time", alpha_series),
            "beta":    ("time", beta_series),
            "lambda":  ("time", lambda_series),   # posterior mean
        },
        coords={"time": time}
    )

    # -------------------------------
    # OPTIONAL PLOTTING
    # Uses updated Gamma-latent plotting function
    # -------------------------------
    if plot:
        plot_alpha_beta_z(
            latent_ds["alpha"].values,
            latent_ds["beta"].values,
            latent_ds["latent"].values,
            title_prefix="MIMO-VAE (Gamma Latent)"
        )

    # -------------------------------
    # OPTIONAL SAVE TO A SINGLE CSV + NC FILES
    # -------------------------------
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save=True")

        latent_df = latent_ds.to_dataframe().reset_index()
        latent_df.to_csv(f"{save_path}_latent_full.csv", index=False)

        sst_ds.to_netcdf(f"{save_path}_sst.nc")
        precip_ds.to_netcdf(f"{save_path}_precip.nc")

    # -------------------------------
    # RETURN EVERYTHING
    # -------------------------------
    return {
        "latent": latent_ds,
        "sst": sst_ds,
        "precip": precip_ds
    }
