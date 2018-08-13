#!/usr/bin/env python

import os
os.environ["OMP_NUM_THREADS"] = "2"  # noqa
os.environ["ARTS_BUILD_PATH"] = "/scratch/uni/u237/users/jmrziglod/arts_general/arts/build"  # noqa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from typhon.files import FileSet, NetCDF4
from typhon.geographical import sea_mask
from typhon.plots import binned_statistic, heatmap, styles, worldmap
from typhon.retrieval import SPAREICE
from typhon.utils import Timer, to_array
import xarray as xr

plt.style.use(styles('typhon'))

experiment = "more_hidden_with_elevation"
train = False
print(f"Perform experiment {experiment}")

# We need to train two retrievals: an ice cloud classifier and an IWP regressor
ice_cloud_fields = [
    'mhs_channel2',
    'mhs_channel3',
    'mhs_channel4',
    'mhs_channel5',
    'solar_azimuth_angle',
    'solar_zenith_angle',
    'avhrr_channel2',
    'avhrr_channel3',
    'avhrr_channel5',
    'avhrr_channel2_std',
    'avhrr_channel5_std',
    'avhrr_tir_diff',
    'lat', 'elevation'
]

iwp_fields = [
    "mhs_channel1", "mhs_channel2",
    "mhs_channel3", "mhs_channel4",
    "mhs_channel5", "lat",  "elevation",
    "mhs_scnpos",  "solar_azimuth_angle",  "solar_zenith_angle",
    #  'satellite_azimuth_angle', 'satellite_zenith_angle',
    "avhrr_channel3", "avhrr_channel4", "avhrr_channel5",
    "avhrr_channel4_std",
    "avhrr_tir_diff",
    "sea_mask",
]

os.makedirs(f"experiments/{experiment}", exist_ok=True)

# Create the SPARE-ICE object. It is not trained so far:
spareice = SPAREICE(
    verbose=2, processes=15, sea_mask_file="data/land_water_mask_5min.png",
    elevation_file="data/surface_elevation_1deg.nc",
)

odata = NetCDF4().read("data/spareice_training_data.nc")
odata = odata.dropna(dim="collocation")

# Convert the collocated data to SPARE-ICE compatible data:
tdata = spareice.standardize_collocations(odata)

# We need ice water paths of 0 g/m^2 to train the ice cloud classifier.
# Unfortunately, zero values would not pass our inhomogeneity filter (
# because it would be NaN). Hence, we "mask" them as very small values:
not_null_tdata = tdata[(tdata.iwp_std / 10**tdata.iwp) < 0.40]
tdata = pd.concat([tdata[np.isnan(tdata.iwp)], not_null_tdata])


def balance(data, lat_bin, bin_points):
    bins = np.arange(-90, 90, lat_bin)
    balanced_data = []
    print(f"split into {bins.size} bins")
    rs = np.random.RandomState(1234)
    for bin_left in bins:
        mask = (bin_left <= data.lat) & (data.lat < bin_left + lat_bin)
        current_bin_points = mask.sum()
        if bin_points <= current_bin_points:
            wanted_bin_points = bin_points
        else:
            wanted_bin_points = current_bin_points

        bin_points_indices = mask.nonzero()[0]

        # We want to have more extreme viewing angles, so they are
        # not underrepresented.
        probability = (np.abs(
            data.mhs_scnpos.iloc[bin_points_indices] - 45) + 0.5) / 50
        probability /= probability.sum()

        print(f"Reduce {current_bin_points} to {wanted_bin_points}")
        selected_indices = rs.choice(
            bin_points_indices, wanted_bin_points, replace=False,
            p=probability
        )
        balanced_data.append(data.iloc[selected_indices])

    return pd.concat(balanced_data)


bin_width = 15
not_null = balance(tdata.dropna(), bin_width, 30_000)
null = balance(tdata[np.isnan(tdata.iwp)], bin_width, 35_000)
bdata = pd.concat([not_null, null])

scat = heatmap(
    not_null.mhs_scnpos,
    not_null.lat, range=[[0, 90], [-90, 90]], bisectrix=False,
    cmap="density", vmin=1,
)
scat.cmap.set_under("w")
plt.colorbar(scat)
plt.savefig(f"experiments/{experiment}/scnpos_lat_heatmap.png")

test_ratio = 0.25
train_data, test_data = train_test_split(
    bdata, test_size=test_ratio, shuffle=True, random_state=5
)

print(f"Use {int(not_null.lat.size*(1-test_ratio))} points for training")
print(f"Use {int(not_null.lat.size*test_ratio)} points for testing")

# Should we train SPARE-ICE?
if train:
    with Timer("SPARE-ICE training"):
        spareice.train(
            train_data,
            iwp_inputs=iwp_fields,
            ice_cloud_inputs=ice_cloud_fields
        )
    print("Best estimator", spareice.iwp.estimator)
    spareice.save(f"experiments/{experiment}/spareice.json")
    print(f"Testing score: {spareice.score(test_data)}")
else:
    spareice.load(f"experiments/{experiment}/spareice.json")

spareice.report("experiments", experiment, test_data)
