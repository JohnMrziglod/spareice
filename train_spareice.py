#!/usr/bin/env python

import os
os.environ["OMP_NUM_THREADS"] = "2"  # noqa
os.environ["ARTS_BUILD_PATH"] = "/scratch/uni/u237/users/jmrziglod/arts_general/arts/build"  # noqa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from typhon.files import CloudSat, FileSet, NetCDF4
from typhon.plots import binned_statistic, heatmap, styles
from typhon.retrieval import SPAREICE
from typhon.utils import Timer, to_array
from scipy.stats import binned_statistic
from typhon.collocations import Collocations
from typhon.geographical import gridded_mean
import xarray as xr

plt.style.use(styles('typhon'))

experiment = "no_sea_mask"
processes = 10
plot_all = False
train = False
zonal_mean = False
balance_scnpos = False
start = "2008"
end = "2009"

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
    'mhs_diff',
    'lat', 'elevation'
]

iwp_fields = [
    #'mhs_channel1', 'mhs_channel2',
    'mhs_channel3', 'mhs_channel4',
    'mhs_channel5', 'avhrr_tir_diff',
    'lat', 'sea_mask',
    'elevation',
    'mhs_scnpos', 'solar_azimuth_angle', 'solar_zenith_angle',
    'avhrr_channel3', 'avhrr_channel4', 'avhrr_channel5', 'avhrr_channel5_std',
    'mhs_diff',
]

os.makedirs(f"experiments/{experiment}", exist_ok=True)

# Create the SPARE-ICE object. It is not trained so far:
spareice = SPAREICE(
    verbose=2, processes=processes,
    sea_mask_file="data/land_water_mask_5min.png",
    elevation_file="data/surface_elevation_1deg.nc",
)

odata = NetCDF4().read("data/spareice_training_data.nc")
odata = odata.dropna(dim="collocation")

# Convert the collocated data to SPARE-ICE compatible data:
tdata = spareice.standardize_collocations(odata)

# We need ice water paths of 0 g/m^2 to train the ice cloud classifier.
# Unfortunately, zero values would not pass our inhomogeneity filter (
# because it would be NaN). Hence, we "mask" them as very small values:
not_null_tdata = tdata[(tdata.iwp_std / 10**tdata.iwp) < 0.5]
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
            p=probability if balance_scnpos else None,
        )
        balanced_data.append(data.iloc[selected_indices])

    return pd.concat(balanced_data)


bin_width = 15
not_null = balance(tdata.dropna(), bin_width, 40_000)
null = balance(tdata[np.isnan(tdata.iwp)], bin_width, 40_000)
bdata = pd.concat([not_null, null])

scat = heatmap(
    not_null.mhs_scnpos,
    not_null.lat, range=[[0, 90], [-90, 90]], bisectrix=False,
    cmap="density", vmin=1,
)
scat.cmap.set_under("w")
plt.colorbar(scat)
plt.savefig(f"experiments/{experiment}/scnpos_lat_heatmap.png")

test_ratio = 0.3
train_data, test_data = train_test_split(
    bdata, test_size=test_ratio, shuffle=True, random_state=5
)

print(f"Use {int(not_null.lat.size*(1-test_ratio))} points for training")
print(f"Use {int(not_null.lat.size*test_ratio)} points for testing")

experiments = FileSet("experiments/{experiment}/spareice.json")

if plot_all:
    spareice = SPAREICE(
        verbose=2, processes=processes,
        sea_mask_file="data/land_water_mask_5min.png",
        elevation_file="data/surface_elevation_1deg.nc",
    )
    for parameters in experiments:
        try:
            print(f"plot experiment {parameters.attr['experiment']}")
            spareice.load(parameters)
            spareice.report(
                "experiments", parameters.attr["experiment"], test_data
            )
        except:
            pass

    exit()

# Should we train SPARE-ICE?
if train:
    with Timer("SPARE-ICE training"):
        spareice.train(
            train_data,
            iwp_inputs=iwp_fields,
            ice_cloud_inputs=ice_cloud_fields,
        )
    print("Best estimator", spareice.iwp.estimator)
    spareice.save(f"experiments/{experiment}/spareice.json")
    print(f"Testing score: {spareice.score(test_data)}")
else:
    spareice.load(f"experiments/{experiment}/spareice.json")

spareice.report("experiments", experiment, test_data)

if not zonal_mean:
    exit()
print(f"Check for mid-latitude bias")


def get_gridded_mean(data, file, spareice):
    retrieved = SPAREICE._retrieve_from_collocations(data, None, spareice)
    if retrieved is None:
        return None

    data = retrieved.dropna(dim="index")

    print(f"Gridding {file.times}")

    lon_bins = np.arange(-180, 185, 5)
    lat_bins = np.arange(-90, 95, 5)

    grid = gridded_mean(
        data.lat.values, data.lon.values, data.iwp.values, (lat_bins, lon_bins)
    )

    return xr.Dataset({
        "IWP_mean": (("lat", "lon"), grid[0]),
        "IWP_number": (("lat", "lon"), grid[1]),
        "lat": lat_bins[:-1],
        "lon": lon_bins[:-1],
    })


def retrieve_spareice(experiment):
    collocations = Collocations(
        path="/work/um0878/user_data/jmrziglod/collocations/MHS_AVHRR/noaa18/"
             "{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}-"
             "{end_hour}{end_minute}{end_second}.nc",
        reference="MHS",
    )

    spareice = SPAREICE(
        file=f"experiments/{experiment}/spareice.json",
        verbose=2, sea_mask_file="data/land_water_mask_5min.png",
        elevation_file="data/surface_elevation_1deg.nc",
    )

    return collocations.map(
        get_gridded_mean, kwargs={
            "spareice": spareice,
        }, on_content=True, pass_info=True, start=start, end=end,
        max_workers=processes, worker_type="process"
    )


def collect_cloudsat():
    cloudsat_files = FileSet(
        name="2C-ICE",
        path="/work/um0878/data/cloudsat/2C-ICE.P1_R04/{year}/{doy}/"
             "{year}{doy}{hour}{minute}{second}_*.hdf.zip",
        handler=CloudSat(),
        # Each file of CloudSat covers exactly 5933 seconds. Since we state it
        # here, the indexing of files is much faster
        time_coverage="5933 seconds",
        # Load only the fields that we need:
        read_args={
            "fields": ["ice_water_path"],
        },
        max_threads=15,
    )

    print("Collect 2C-ICE...")
    data = xr.concat(
        cloudsat_files[start:end],
        dim="scnline"
    )

    data.to_netcdf(f"data/2C-ICE_{start}.nc")
    return data


def plot_zonal_mean(ax, lat, iwp, label):
    lat_bins = np.arange(-90, 85, 5)
    zonal, _, _ = binned_statistic(
        lat.values, iwp.values, np.nanmean, bins=lat_bins
    )
    print(label, zonal)
    ax.plot(lat_bins[:-1] + 2.5, zonal, label=label)


cpr = xr.open_dataset("data/2C-ICE_gridded_2008.nc")
cpr.load()
cgridded = cpr["2C-ICE_mean"].values

result_list = [r for r in retrieve_spareice(experiment) if r is not None]
results = xr.concat(result_list, dim="time")
weights = results.IWP_number / results.IWP_number.sum("time")
gridded = (weights * results.IWP_mean).sum("time")

print(gridded.values.mean(axis=1))
print(gridded - cgridded)

fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(results.lat.values+2.5, cgridded.mean(axis=1), label="2C-ICE")
ax.plot(results.lat.values+2.5, gridded.values.mean(axis=1),
        label=f"SPARE-ICE ({experiment})")
ax.legend()
fig.savefig(f"experiments/{experiment}/zonal_mean_{start}.png")

with open(f"experiments/{experiment}/stats.txt", "w") as file:
    file.write(f"{gridded.values.mean(axis=1)}")


gridded.to_netcdf(f"data/{experiment}/SPARE-ICE_gridded_{start}.nc")
