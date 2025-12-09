"""
Imports
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
    Derived Gamma mean/std:
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
    pdf = PdfPages(f"{file_name}.pdf")

    # unpack (Gamma latent)
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
    """
    Convenience wrapper: tensor (torch) -> numpy via scaler.inverse_transform.
    """
    return scaler.inverse_transform(tensor.detach().cpu().numpy())


"""
Dataset
"""

class PairedAnomalyDataset(Dataset):
    def __init__(self, sst_tensor, precip_tensor):
        assert sst_tensor.shape[0] == precip_tensor.shape[0], "Time dimension mismatch"
        self.sst_tensor = sst_tensor
        self.precip_tensor = precip_tensor

    def __len__(self):
        return self.sst_tensor.shape[0]

    def __getitem__(self, idx):
        return self.sst_tensor[idx], self.precip_tensor[idx]


"""
MIMOVAE with Gamma latent and GNLL(Gaussian) for SST
"""

class MIMOVAE(nn.Module):
    def __init__(self, sst_dim, precip_dim):
        super(MIMOVAE, self).__init__()

        # --- Encoders ---
        self.encoder_sst = nn.Sequential(
            nn.Linear(sst_dim, 50), nn.Tanh(),
            nn.Linear(50, 10), nn.Tanh()
        )

        self.encoder_precip = nn.Sequential(
            nn.Linear(precip_dim, 50), nn.Tanh(),
            nn.Linear(50, 10), nn.Tanh()
        )

        # Shared
        self.shared_hidden = nn.Linear(20, 10)

        # Gamma posterior q(z|x) = Gamma(alpha, beta)
        self.alpha_head = nn.Sequential(nn.Linear(10, 1), nn.Softplus())
        self.beta_head  = nn.Sequential(nn.Linear(10, 1), nn.Softplus())

        # SST decoder trunk + heads for mean/variance
        self.decoder_sst_trunk = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(),
            nn.Linear(10, 50), nn.Tanh()
        )
        self.sst_mu_head = nn.Linear(50, sst_dim)
        self.sst_logvar_head = nn.Linear(50, sst_dim)

        # Precipitation decoder: predicts mean in scaled space
        self.decoder_precip = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(),
            nn.Linear(10, 50), nn.Tanh(),
            nn.Linear(50, precip_dim)
        )

    def reparameterize(self, alpha, beta):
        dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
        return dist.rsample()

    def forward(self, sst_x, precip_x):
        # Encode
        sst_encoded = self.encoder_sst(sst_x)
        precip_encoded = self.encoder_precip(precip_x)

        # Shared representation
        h = self.shared_hidden(torch.cat([sst_encoded, precip_encoded], dim=1))

        # Gamma parameters (posterior)
        alpha = self.alpha_head(h) + 1e-4
        beta  = self.beta_head(h)  + 1e-4

        # Latent sample
        z = self.reparameterize(alpha, beta)

        # SST outputs: μ and σ²
        sst_hidden = self.decoder_sst_trunk(z)
        sst_mu = self.sst_mu_head(sst_hidden)
        sst_logvar = self.sst_logvar_head(sst_hidden)
        sst_var = torch.nn.functional.softplus(sst_logvar) + 1e-6

        # Precip: mean in scaled space
        precip_mu = torch.nn.functional.softplus(self.decoder_precip(z)) + 1e-6

        return sst_mu, sst_var, precip_mu, alpha, beta, z



"""
Training function
"""

def learn(
    sst_dat,
    precip_dat,
    norm,
    sst_var='sst',
    precip_var='tp',
    gshape=1.0,
    grate=1.0,
    train_pct=0.8,
    batch=32,
    epochs=100,
    verbose=True
):
    """
    Train MIMOVAE with:
      - Gaussian NLL reconstruction loss for SST (in scaled space)
      - Gamma NLL reconstruction loss for precipitation (in scaled space, MinMax)
      - KL divergence between Gamma posterior and Gamma prior in latent space

    `norm` controls SST scaling:
        'standard' -> StandardScaler for SST
        'minmax'   -> MinMaxScaler for SST

    Precipitation is ALWAYS MinMax-scaled (0..1) to ensure positivity for Gamma NLL.
    """

    # -------------------------------
    # DATA PREPARATION
    # -------------------------------
    sst_np = np.nan_to_num(sst_dat[sst_var].values)
    precip_np = np.nan_to_num(precip_dat[precip_var].values)

    # Flatten spatial dimensions
    sst_flat = sst_np.reshape(sst_np.shape[0], -1)
    precip_flat = precip_np.reshape(precip_np.shape[0], -1)

    # SST scaling
    if norm == 'standard':
        sst_scaler = StandardScaler()
    elif norm == 'minmax':
        sst_scaler = MinMaxScaler()
    else:
        raise ValueError("norm must be 'standard' or 'minmax'")

    # Precipitation: always MinMax to keep values in [0, 1]
    precip_scaler = MinMaxScaler()

    sst_scaled = sst_scaler.fit_transform(sst_flat)
    precip_scaled = precip_scaler.fit_transform(precip_flat)

    # Safety check for precip positivity
    if np.any(precip_scaled <= 0.0):
        # This should rarely happen with MinMaxScaler, but we guard anyway
        precip_scaled = np.clip(precip_scaled, 1e-6, None)

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

    # Gaussian NLL for SST (scaled space)
    gaussian_nll = torch.nn.GaussianNLLLoss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # tracking
    losses, precip_losses, sst_losses = [], [], []
    nd_precip_losses, nd_sst_losses = [], []

    # -------------------------------
    # RECONSTRUCTION: Gamma NLL for precipitation (scaled)
    # -------------------------------
    def gamma_nll(y_true, y_pred_mean, shape=0.5, eps=1e-6):
        """
        Gamma negative log-likelihood in shape–rate parameterization,
        using decoder-predicted mean and fixed shape.
        """
        y_true = torch.clamp(y_true, min=eps)
        y_pred_mean = torch.clamp(y_pred_mean, min=eps)

        shape_t = torch.as_tensor(shape, dtype=y_true.dtype, device=y_true.device)
        rate = shape_t / (y_pred_mean + eps)  # β = k / μ

        nll = (
            torch.lgamma(shape_t)
            - shape_t * torch.log(rate)
            + (shape_t - 1.0) * torch.log(y_true)
            + rate * y_true
        )
        return nll.mean()

    # -------------------------------
    # KL TERM: Gamma posterior vs Gamma prior
    # -------------------------------
    prior_alpha = torch.tensor(gshape, dtype=torch.float32, device=device)  # shape
    prior_beta = torch.tensor(grate, dtype=torch.float32, device=device)    # rate

    def gamma_kl(alpha_q, beta_q, alpha_p, beta_p):
        """
        KL( Gamma(alpha_q, beta_q) || Gamma(alpha_p, beta_p) )
        shape-rate parameterization.
        """
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
        model.train()
        for sst_batch, precip_batch in train_loader:
            sst_batch = sst_batch.to(device)
            precip_batch = precip_batch.to(device)

            # Forward pass:
            # model returns: sst_mu, sst_var, precip_mu, alpha, beta, z
            sst_mu, sst_var, precip_mu, alpha, beta, z = model(sst_batch, precip_batch)

            # Reconstruction losses (scaled space)
            loss_sst = gaussian_nll(
                input=sst_mu,
                target=sst_batch,
                var=sst_var
            )

            loss_precip = gamma_nll(
                y_true=precip_batch,
                y_pred_mean=precip_mu,
                shape=gshape
            )

            # KL divergence in latent space
            kl = kl_loss(alpha, beta)

            # Total loss
            loss = loss_sst + loss_precip + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ------------------------------------------------------------------
            # Physical-space diagnostics (MSE after inverse scaling)
            # ------------------------------------------------------------------
            sst_loss = np.mean(
                (inverse_transform(sst_scaler, sst_mu) -
                 inverse_transform(sst_scaler, sst_batch)) ** 2
            )
            precip_loss = np.mean(
                (inverse_transform(precip_scaler, precip_mu) -
                 inverse_transform(precip_scaler, precip_batch)) ** 2
            )

            losses.append(loss.item())
            nd_sst_losses.append(loss_sst.item())
            nd_precip_losses.append(loss_precip.item())
            sst_losses.append(sst_loss)
            precip_losses.append(precip_loss)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Total: {loss.item():.4f} | "
                f"SST (GNLL): {loss_sst.item():.4f} | "
                f"Precip (Gamma NLL): {loss_precip.item():.4f} | "
                f"KL: {kl.item():.4f}"
            )

    # -------------------------------
    # LATENT EXTRACTION (alpha, beta, z) ON TEST SET
    # -------------------------------
    model.eval()
    alpha_list, beta_list, z_list = [], [], []

    with torch.no_grad():
        for sst_batch, precip_batch in test_loader:
            sst_batch = sst_batch.to(device)
            precip_batch = precip_batch.to(device)

            _, _, _, alpha, beta, z = model(sst_batch, precip_batch)
            alpha_list.append(alpha.cpu().numpy())
            beta_list.append(beta.cpu().numpy())
            z_list.append(z.cpu().numpy())

    alpha_series = np.concatenate(alpha_list, axis=0)
    beta_series = np.concatenate(beta_list, axis=0)
    z_series = np.concatenate(z_list, axis=0)

    # -------------------------------
    # TEST LOSSES (physical space MSE)
    # -------------------------------
    mse_phys = nn.MSELoss(reduction='mean')

    test_sst_losses, test_precip_losses = [], []

    with torch.no_grad():
        for sst_batch, precip_batch in test_loader:
            sst_batch = sst_batch.to(device)
            precip_batch = precip_batch.to(device)

            sst_mu, sst_var, precip_mu, alpha, beta, z = model(sst_batch, precip_batch)

            # Back to physical space for diagnostics
            sst_mu_phys = torch.tensor(
                inverse_transform(sst_scaler, sst_mu),
                dtype=torch.float32,
                device=device
            )
            sst_batch_phys = torch.tensor(
                inverse_transform(sst_scaler, sst_batch),
                dtype=torch.float32,
                device=device
            )

            precip_mu_phys = torch.tensor(
                inverse_transform(precip_scaler, precip_mu),
                dtype=torch.float32,
                device=device
            )
            precip_batch_phys = torch.tensor(
                inverse_transform(precip_scaler, precip_batch),
                dtype=torch.float32,
                device=device
            )

            test_sst_loss = mse_phys(sst_mu_phys, sst_batch_phys).item()
            test_precip_loss = mse_phys(precip_mu_phys, precip_batch_phys).item()

            test_sst_losses.append(test_sst_loss)
            test_precip_losses.append(test_precip_loss)

    # -------------------------------
    # RETURN EVERYTHING
    # -------------------------------
    return {
        "model": model,
        "sst_scaler": sst_scaler,
        "precip_scaler": precip_scaler,
        "losses": losses,
        "nd_sst_losses": nd_sst_losses,       # GNLL on scaled SST
        "nd_precip_losses": nd_precip_losses, # Gamma NLL on scaled precip
        "sst_losses": sst_losses,             # physical-space MSE (train)
        "precip_losses": precip_losses,       # physical-space MSE (train)
        "test_sst_losses": test_sst_losses,   # physical-space MSE (test)
        "test_precip_losses": test_precip_losses,
        "alpha_series": alpha_series,
        "beta_series": beta_series,
        "z_series": z_series,
        "train_len": train_len
    }


"""
Extraction function
"""

def extractr(model, sst_dat, svar, precip_dat, pvar, sst_scaler, precip_scaler,
             plot=True, save=False, save_path=None):
    """
    Run the trained MIMOVAE over the full time series and return:
      - reconstructed SST / precip fields in physical space
      - latent (z, alpha, beta, lambda) as an xarray.Dataset
    """
    import numpy as np
    import torch
    import xarray as xr

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
    # (returns: sst_mu, sst_var, precip_mu, alpha, beta, z)
    # -------------------------------
    model.eval()
    with torch.no_grad():
        sst_mu, sst_var, precip_mu, alpha, beta, z = model(sst_tensor, precip_tensor)

        # convert tensors → numpy
        sst_mu = sst_mu.cpu().numpy()
        precip_mu = precip_mu.cpu().numpy()
        alpha_series = alpha.cpu().numpy().squeeze()
        beta_series  = beta.cpu().numpy().squeeze()
        z_series     = z.cpu().numpy().squeeze()

    # -------------------------------
    # INVERSE TRANSFORM RECONSTRUCTIONS
    # (Use posterior mean predictions)
    # -------------------------------
    sst_recon = sst_scaler.inverse_transform(sst_mu).reshape(sst_np.shape)
    precip_recon = precip_scaler.inverse_transform(precip_mu).reshape(precip_np.shape)

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
    # COMPUTE POSTERIOR MEAN OF LATENT GAMMA VARIABLE
    # lambda = E[z|x] = alpha / beta
    # -------------------------------
    lambda_series = alpha_series / beta_series

    latent_ds = xr.Dataset(
        {
            "latent":  ("time", z_series),
            "alpha":   ("time", alpha_series),
            "beta":    ("time", beta_series),
            "lambda":  ("time", lambda_series),
        },
        coords={"time": time}
    )

    # -------------------------------
    # OPTIONAL PLOTTING
    # -------------------------------
    if plot:
        plot_alpha_beta_z(
            latent_ds["alpha"].values,
            latent_ds["beta"].values,
            latent_ds["latent"].values,
            title_prefix="MIMO-VAE (Gamma Latent)"
        )

    # -------------------------------
    # OPTIONAL SAVE OUTPUTS
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
