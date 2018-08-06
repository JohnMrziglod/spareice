import os
os.environ["OMP_NUM_THREADS"] = "2"  # noqa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neural_network as nn
from typhon.files import FileSet, NetCDF4
from typhon.plots import binned_statistic, heatmap, styles, worldmap
from typhon.retrieval import SPAREICE
import xarray as xr

plt.style.use(styles('typhon'))

rdata = NetCDF4().read("spareice_training_data.nc")
rdata = rdata.dropna(dim="collocation")

# Filter out the IWPs that are 0 g/m^2.
tdata = rdata.isel(
    collocation=rdata["MHS_2C-ICE/2C-ICE/ice_water_path_mean"].values > 10e-6)


def balance(data, lat_bin, bin_points):
    bins = np.arange(-90, 90, lat_bin)
    print(bins)
    balanced_data = []
    print(f"split into {bins.size} bins")
    for bin_left in bins:
        mask = (bin_left <= data.lat.values) & (
                    data.lat.values < bin_left + lat_bin)
        current_bin_points = mask.sum()
        wanted_bin_points = bin_points if bin_points <= current_bin_points else current_bin_points
        print(f"Reduce {current_bin_points} to {wanted_bin_points}")
        indices = np.random.choice(mask.nonzero()[0], wanted_bin_points,
                                   replace=False)
        balanced_data.append(data.isel(collocation=indices))

    return xr.concat(balanced_data, dim="collocation")


bin_width = 15
bdata = balance(tdata, bin_width, 13_000)
bdata.lat.plot.hist(bins=180 // (bin_width), range=[-90, 90])

heatmap(bdata.)

# Should we train SPARE-ICE?
train_data, test_data = SPAREICE.split_data(bdata)

if True:
    spareice = SPAREICE(verbose=0, processes=15)
    spareice.train(train_data)
    spareice.save_training("spareice.json")
    print(f"Testing score: {spareice.score(test_data)}")
else:
    spareice = SPAREICE(file="spareice.json", verbose=2)

spareice_iwp = spareice.retrieve(
    test_data, from_collocations=True, as_log10=True
)
test_iwp = spareice.get_targets(test_data)

print("Plot test results")
fig, ax = plt.subplots(figsize=(10, 8))
scat = heatmap(
    test_iwp["iwp_log10"].values,
    spareice_iwp["iwp_log10"].values,
    bins=50, range=[[-1, 4], [-1, 4]],
    cmap="density"
)
ax.set_xlabel("log10 IWP (2C-ICE) [g/m^2]")
ax.set_ylabel("log10 IWP (SPARE-ICE) [g/m^2]")
fig.colorbar(scat)
fig.savefig("plots/2C-ICE-SPAREICE_heatmap.png")

fig, ax = plt.subplots(figsize=(10, 8))
fe = 100 * (np.exp(np.abs(np.log(
    10 ** spareice_iwp["iwp_log10"].values / 10 ** test_iwp[
        "iwp_log10"].values))) - 1)
binned_statistic(
    test_iwp["iwp_log10"].values,
    fe, range=[-1, 4], pargs={"marker": "o"}
)
ax.set_xlabel("log10 IWP (2C-ICE) [g/m^2]")
ax.set_ylabel("Median fractional error [%]")
ax.set_ylim([0, 400])
fig.savefig("plots/2C-ICE-SPAREICE_mfe.png")

fig, ax = plt.subplots(figsize=(10, 8))
binned_statistic(
    test_data.lat.values,
    fe, pargs={"marker": "o"}
)
ax.set_xlabel("latitude")
ax.set_ylabel("Median fractional error [%]")
ax.set_ylim([0, 200])

fig, ax = plt.subplots(figsize=(10, 8))
scat = plt.scatter(
    test_iwp["iwp_log10"].values,
    spareice_iwp["iwp_log10"].values,
    s=1, alpha=0.3
)
ax.set_xlabel("log10 IWP (2C-ICE) [g/m^2]")
ax.set_ylabel("log10 IWP (SPARE-ICE) [g/m^2]")
fig.savefig("plots/2C-ICE-SPAREICE_scatter.png")

fig, ax = plt.subplots(figsize=(10, 8))
bias = 10 ** spareice_iwp["iwp_log10"].values - 10 ** test_iwp[
    "iwp_log10"].values
binned_statistic(
    test_iwp["iwp_log10"].values,
    bias, range=[-1, 4], pargs={"marker": "o"}
)
ax.set_xlabel("log10 IWP (2C-ICE) [g/m^2]")
ax.set_ylabel("IWP (SPARE-ICE - 2C-ICE) [g/m^2]")
# ax.set_ylim([0, 400])
fig.savefig("plots/2C-ICE-SPAREICE_bias.png")

fig, ax = plt.subplots(figsize=(10, 8))
binned_statistic(
    test_data.lat.values,
    bias, pargs={"marker": "o"}
)
ax.set_xlabel("latitude")
ax.set_ylabel("IWP (SPARE-ICE - 2C-ICE) [g/m^2]")
# ax.set_ylim([0, 200])
fig.savefig("plots/2C-ICE-SPAREICE_bias_lat.png")