import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def ncKDE(var: str, model:str, path: str, log:bool = None):
    sns.set_theme(style = 'white')
    prdata = xr.open_dataset(f"~/{path}")
    precip = prdata[f"{var}"].values  # shape (time, lat, lon)
    precip_flat = precip.flatten()
    
    # Remove invalids
    precip_flat = precip_flat[~np.isnan(precip_flat)]
    if log == True:
         plotdat = np.log1p(precip_flat)  # log(1 + precip)
    else: plotdat = precip_flat
        
    plt.figure(figsize=(8,5))
    sns.kdeplot(plotdat, fill=True)
    plt.xlabel(f"{var}")
    plt.title(f"KDE of {model} ({var})")
    plt.show()

def latentKDE( var: str, model:str, path: str):
    sns.set_theme(style = 'white')
    latent = pd.read_csv(f"~/{path}")
    plt.figure(figsize=(8,5))
    sns.kdeplot(latent[var].values, fill=True)
    plt.xlabel(f"{var}")
    plt.title(f"KDE of {model} Latent Index ({var})")
    plt.show()

def csvKDE(var: str, model: str, df, log: bool = False):
    sns.set_theme(style='white')

    # If the user passed a string, treat it as a CSV filepath
    if isinstance(df, str):
        df = pd.read_csv(f"{df}")   # do not prepend "~/" automatically
                                     # let the shell or user control paths

    # If the DataFrame has multiple columns, you must specify which one
    if df.shape[1] != 1:
        raise ValueError(
            "DataFrame must contain exactly one column for KDE, "
            "or modify the function to select a column."
        )

    data = df.iloc[:, 0].values

    if log:
        data = np.log(data)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data, fill=True)
    plt.xlabel(var)
    plt.title(f"KDE of {model} ({var})")
    plt.show()


def universalKDE(var: str, model: str, data, log: bool = False):
    """
    Plot a KDE for:
      - CSV file path
      - pandas DataFrame or Series
      - xarray DataArray
      - xarray Dataset (single-variable or user-specified via 'var')
    """

    sns.set_theme(style='white')

    # ---------------------------------------------------------------------
    # CASE 1: DATA IS A STRING → treat as file path to CSV
    # ---------------------------------------------------------------------
    if isinstance(data, str):
        path = Path(data).expanduser()
        df = pd.read_csv(path)

        if df.shape[1] != 1:
            raise ValueError(
                f"CSV contains multiple columns; KDE needs one column. "
                f"Select a column or adjust function. Columns: {df.columns.tolist()}"
            )

        values = df.iloc[:, 0].values

    # ---------------------------------------------------------------------
    # CASE 2: DATA IS A PANDAS DATAFRAME / SERIES
    # ---------------------------------------------------------------------
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(
                f"DataFrame contains multiple columns; KDE needs one column. "
                f"Columns: {data.columns.tolist()}"
            )
        values = data.iloc[:, 0].values

    elif isinstance(data, pd.Series):
        values = data.values

    # ---------------------------------------------------------------------
    # CASE 3: DATA IS AN XARRAY DATAARRAY
    # ---------------------------------------------------------------------
    elif isinstance(data, xr.DataArray):
        values = data.values.flatten()

    # ---------------------------------------------------------------------
    # CASE 4: DATA IS AN XARRAY DATASET
    # ---------------------------------------------------------------------
    elif isinstance(data, xr.Dataset):
        # If variable name is provided, use it
        if var in data:
            da = data[var]
        else:
            # Try auto-selection if only one variable exists
            vars = list(data.data_vars)
            if len(vars) == 1:
                da = data[vars[0]]
            else:
                raise ValueError(
                    f"Dataset contains multiple variables but '{var}' is not one of them. "
                    f"Available: {vars}"
                )

        values = da.values.flatten()

    else:
        raise TypeError(
            "Input must be one of: str (CSV path), DataFrame, Series, "
            "xarray.DataArray, or xarray.Dataset."
        )

    # ---------------------------------------------------------------------
    # LOG TRANSFORM
    # ---------------------------------------------------------------------
    if log:
        values = np.log(values)

    # ---------------------------------------------------------------------
    # PLOT KDE
    # ---------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.kdeplot(values, fill=True)
    plt.xlabel(var)
    plt.title(f"KDE of {model} ({var})")
    plt.show()

def standardizeNC(path: str, feature_dim: str = None, vars_to_scale: list = None,) -> xr.Dataset:
    """
    Standardize variables in a NetCDF file using sklearn's StandardScaler.

    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    vars_to_scale : list, optional
        List of variable names to scale. If None, all numeric variables are scaled.
    feature_dim : str, optional
        Dimension along which observations lie (e.g., "time"). If None,
        the function attempts to infer a single feature dimension.

    Returns
    -------
    xr.Dataset
        Dataset with standardized variables.
    """
    
    ds = xr.open_dataset(path)

    # Select variables to scale
    if vars_to_scale is None:
        vars_to_scale = [
            v for v in ds.data_vars
            if np.issubdtype(ds[v].dtype, np.number)
        ]

    # Infer feature dimension if not provided
    if feature_dim is None:
        # Common default choice is "time"
        if "time" in ds.dims:
            feature_dim = "time"
        else:
            # Try to find any dimension with length > 1
            candidates = [dim for dim in ds.dims if ds.dims[dim] > 1]
            if len(candidates) != 1:
                raise ValueError(
                    "Cannot infer feature dimension automatically. "
                    "Please specify `feature_dim` explicitly."
                )
            feature_dim = candidates[0]

    scaler = StandardScaler()
    scaled_data = {}

    for var in vars_to_scale:
        arr = ds[var]

        # Move feature_dim to front for reshaping
        arr_transposed = arr.transpose(feature_dim, ...)
        original_shape = arr_transposed.shape

        # Flatten all non-feature dims for sklearn
        reshaped = arr_transposed.values.reshape(original_shape[0], -1)

        # Fit and transform
        scaled_values = scaler.fit_transform(reshaped)

        # Reshape back to original
        scaled_values = scaled_values.reshape(original_shape)

        # Return to original dimension order
        scaled_da = xr.DataArray(
            data=scaled_values.transpose(arr.get_axis_num(feature_dim), *[
                arr.get_axis_num(d) for d in arr.dims if d != feature_dim
            ]),
            coords=arr.coords,
            dims=arr.dims,
            attrs=arr.attrs
        )

        scaled_data[var] = scaled_da

    # Build dataset
    scaled_ds = ds.copy()
    for var, da in scaled_data.items():
        scaled_ds[var] = da

    return scaled_ds
