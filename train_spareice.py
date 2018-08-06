#!/usr/bin/env python

import os
os.environ["OMP_NUM_THREADS"] = "2"  # noqa
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neural_network as nn
from typhon.files import FileSet, NetCDF4
from typhon.geographical import sea_mask
from typhon.plots import binned_statistic, heatmap, styles
from typhon.retrieval import SPAREICE
from typhon.utils import Timer
import xarray as xr

plt.style.use(styles('typhon'))

experiment = "new_alpha_values"
print(f"Perform experiment {experiment}")

training_fields = [
    "mhs_channel1", "mhs_channel2", "mhs_channel3", "mhs_channel4",
    "mhs_channel5",
    "cloud_filter", "lat",
    "sea_mask",
    "mhs_scnpos",
    "solar_azimuth_angle", "solar_zenith_angle",
    "avhrr_channel3", "avhrr_channel4",
    "avhrr_channel4_std",
    "avhrr_channel5"
]

os.makedirs(f"experiments/{experiment}", exist_ok=True)

odata = NetCDF4().read("spareice_training_data.nc")
odata = odata.dropna(dim="collocation")

good = odata["MHS_2C-ICE/2C-ICE/ice_water_path_number"] > 11
good &= odata["MHS_2C-ICE/2C-ICE/ice_water_path_std"] \
        / odata["MHS_2C-ICE/2C-ICE/ice_water_path_mean"] < 0.5
tdata = odata.sel(collocation=good)
tdata["sea_mask"] = "collocation", sea_mask(
    tdata.lat, tdata.lon, "land_water_mask_5min.png"
)


def balance(data, lat_bin, bin_points):
    bins = np.arange(-90, 90, lat_bin)
    balanced_data = []
    rest_data = []
    r = np.random.RandomState(1234)
    for bin_left in bins:
        mask = (bin_left <= data.lat.values) & (
                    data.lat.values < bin_left + lat_bin)
        current_bin_points = mask.sum()
        if bin_points <= current_bin_points:
            wanted_bin_points = bin_points
        else:
            wanted_bin_points = current_bin_points
        bin_points_indices = mask.nonzero()[0]
        print(f"Reduce {current_bin_points} to {wanted_bin_points}")
        selected_indices = r.choice(bin_points_indices,
                                    wanted_bin_points, replace=False)
        balanced_data.append(data.isel(collocation=selected_indices))
        rest_indices = np.setdiff1d(np.arange(wanted_bin_points),
                                    selected_indices, assume_unique=True)
        rest_data.append(data.isel(collocation=rest_indices))

    return xr.concat(balanced_data, dim="collocation"), \
        xr.concat(rest_data, dim="collocation")


def plot_scatter(xdata, ydata, sea_mask):
    for area in ["all", "land", "sea"]:
        if area == "all":
            mask = slice(None, None, None)
        elif area == "land":
            mask = ~sea_mask
        else:
            mask = sea_mask

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            xdata[mask], ydata[mask],
            s=1, alpha=0.6
        )
        ax.grid()
        ax.set_xlabel("log10 IWP (2C-ICE) [g/m^2]")
        ax.set_ylabel("log10 IWP (SPARE-ICE) [g/m^2]")
        ax.set_title(f"{experiment} - {area}")
        fig.savefig(
            f"experiments/{experiment}/2C-ICE-SPAREICE_scatter_{area}.png"
        )


def plot_error(xdata, error, sea_mask, file, on_lat=False, mfe=True):

    fig, ax = plt.subplots(figsize=(10, 8))
    if on_lat:
        xlabel = "latitude"
        xrange = [-90, 90]
    else:
        xlabel = "log10 IWP (2C-ICE) [g/m^2]"
        xrange = [-1, 4]

    if mfe:
        ax.set_ylabel("Median fractional error [%]")
        ax.set_ylim([0, 400])
        statistic = "median"
    else:
        ax.set_ylabel("$\Delta$ IWP (SPARE-ICE - 2C-ICE) [log 10 g/m^2]")
        statistic = "mean"

    for area in ["all", "land", "sea"]:
        if area == "all":
            mask = slice(None, None, None)
        elif area == "land":
            mask = ~sea_mask
        else:
            mask = sea_mask

        binned_statistic(
            xdata[mask], error[mask], statistic=statistic,
            range=xrange, pargs={"marker": "o", "label": area}
        )

    ax.set_xlabel(xlabel)
    ax.grid()
    ax.legend(fancybox=True)
    ax.set_title(f"Experiment: {experiment}")
    fig.tight_layout()
    fig.savefig(f"experiments/{experiment}/{file}")


bin_width = 15
bdata, rdata = balance(tdata, bin_width, 20_000)

scat = heatmap(
    bdata["MHS_2C-ICE/MHS/scnpos"].values,
    bdata.lat.values, range=[[0, 90], [-90, 90]], bisectrix=False,
    cmap="density", vmin=1,
)
scat.cmap.set_under("w")
plt.colorbar(scat)
plt.savefig(f"experiments/{experiment}/scnpos_lat_heatmap.png")

# Should we train SPARE-ICE?
train_data, test_data = SPAREICE.split_data(bdata)
#test_data = xr.concat([test_data, rdata], dim="collocation")

if True:
    spareice = SPAREICE(verbose=2, processes=15)
    with Timer("SPARE-ICE training"):
        spareice.train(train_data, fields=training_fields)
    spareice.save_training(f"experiments/{experiment}/spareice.json")
    print(f"Testing score: {spareice.score(test_data, training_fields)}")
else:
    spareice = SPAREICE(
        file=f"experiments/{experiment}/spareice.json", verbose=2
    )

spareice_iwp = spareice.retrieve(
    test_data, from_collocations=True, as_log10=True
)
test_iwp = spareice.get_targets(test_data)

print("Plot test results")

# HEATMAP
fig, ax = plt.subplots(figsize=(10, 8))
scat = heatmap(
    test_iwp["iwp_log10"].values,
    spareice_iwp["iwp_log10"].values,
    bins=50, range=[[-1, 4], [-1, 4]],
    cmap="density"
)
ax.set_xlabel("log10 IWP (2C-ICE) [g/m^2]")
ax.set_ylabel("log10 IWP (SPARE-ICE) [g/m^2]")
ax.set_title(experiment)
fig.colorbar(scat)
fig.savefig(f"experiments/{experiment}/2C-ICE-SPAREICE_heatmap.png")

plot_scatter(
    test_iwp["iwp_log10"].values,
    spareice_iwp["iwp_log10"].values,
    test_data.sea_mask.values
)


# MFE plot with 2C-ICE on x-axis
fe = 100 * (
    np.exp(np.abs(
        np.log(
            10**spareice_iwp["iwp_log10"].values
            / 10**test_iwp["iwp_log10"].values
        )
    )) - 1
)
plot_error(
    test_iwp["iwp_log10"].values,
    fe, test_data["sea_mask"].values,
    file="2C-ICE-SPAREICE_mfe.png"
)
# MFE plot with latitude on x-axis
plot_error(
    test_data.lat.values,
    fe, test_data["sea_mask"].values,
    file="2C-ICE-SPAREICE_mfe_lat.png",
    on_lat=True
)

# Plot the bias:
bias = spareice_iwp["iwp_log10"].values - test_iwp["iwp_log10"].values
plot_error(
    test_iwp["iwp_log10"].values,
    bias, test_data["sea_mask"].values,
    file="2C-ICE-SPAREICE_bias.png",
    mfe=False,
)
# MFE plot with latitude on x-axis
plot_error(
    test_data.lat.values,
    bias, test_data["sea_mask"].values,
    file="2C-ICE-SPAREICE_bias_lat.png",
    mfe=False, on_lat=True
)

print(f"Experiment {experiment} was successfully performed!")
