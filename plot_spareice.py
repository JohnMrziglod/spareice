#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from typhon.collocations import Collocations
from typhon.files import CloudSat, FileSet
from typhon.plots import styles
from typhon.geographical import gridded_mean
from typhon.retrieval import SPAREICE
import xarray as xr

plt.style.use(styles('typhon'))



VERSION = "best_spareice"
START = "2008"
END = "2009"
PROCESSES = 20

print(f"Plot experiment {VERSION}")


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


def collect_spareice(version):
    spareice_files = FileSet(
       name="SPAREICE",
       path=f"/work/um0878/user_data/jmrziglod/spareice/{version}/noaa18/"
            "{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}-"
            "{end_hour}{end_minute}{end_second}.nc",
       max_processes=PROCESSES,
       placeholder={"version": version}
    )

    print("Collect SPARE-ICE...")
    data_list = spareice_files.map(
        get_gridded_mean, start=START, end=END, on_content=True,
        pass_info=True,
    )
    data = xr.concat(data_list, dim="time")
    #data.to_netcdf(f"data/{version}_SPARE-ICE_{START}.nc")
    return data


def retrieve_spareice(version):
    collocations = Collocations(
        path="/work/um0878/user_data/jmrziglod/collocations/MHS_AVHRR/noaa18/"
             "{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}-"
             "{end_hour}{end_minute}{end_second}.nc",
        reference="MHS",
    )

    spareice = SPAREICE(
        file=f"experiments/{version}/spareice.json",
        verbose=2, sea_mask_file="data/land_water_mask_5min.png",
        elevation_file="data/surface_elevation_1deg.nc",
    )

    return collocations.map(
        get_gridded_mean, kwargs={
            "spareice": spareice,
        }, on_content=True, pass_info=True, start=START, end=END,
        max_workers=PROCESSES, worker_type="process"
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
        cloudsat_files[START:END],
        dim="scnline"
    )

    data.to_netcdf(f"data/2C-ICE_{START}.nc")
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

result_list = [r for r in retrieve_spareice(VERSION) if r is not None]
results = xr.concat(result_list, dim="time")
weights = results.IWP_number / results.IWP_number.sum("time")
gridded = (weights * results.IWP_mean).sum("time")
gridded.to_netcdf(f"data/{VERSION}_SPARE-ICE_{START}.nc")

fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(results.lat.values+2.5, cgridded.mean(axis=1), label="2C-ICE")
ax.plot(results.lat.values+2.5, gridded.values.mean(axis=1),
        label=f"SPARE-ICE ({VERSION})")
ax.legend()
fig.savefig(f"experiments/{VERSION}/zonal_mean_{START}.png")
