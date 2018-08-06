import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from typhon.files import CloudSat, FileSet
from typhon.plots import styles
import xarray as xr

plt.style.use(styles('typhon'))

START = "2007"
END = "2008"


def collect_spareice():
    spareice_files = FileSet(
       name="SPAREICE",
       path="/work/um0878/user_data/jmrziglod/spareice/noaa18/"
            "{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}-"
            "{end_hour}{end_minute}{end_second}.nc",
       max_threads=15,
    )

    print("Collect SPARE-ICE...")
    data = xr.concat(
       spareice_files[START:END],
       dim="collocation"
    )

    data.to_netcdf(f"SPARE-ICE_{START}.nc")
    return data


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
        max_threads=10,
    )

    print("Collect 2C-ICE...")
    data = xr.concat(
        cloudsat_files[START:END],
        dim="scnline"
    )

    data.to_netcdf(f"2C-ICE_{START}.nc")
    return data


def plot_zonal_mean(ax, lat, iwp, label):
    lat_bins = np.arange(-90, 85, 5)
    zonal, _, _ = binned_statistic(
        lat.values, iwp.values, np.nanmean, bins=lat_bins
    )
    print(label, zonal)
    ax.plot(lat_bins[:-1] + 2.5, zonal, label=label)


#cloudsat = collect_cloudsat()
cloudsat = xr.open_dataset(f"2C-ICE_{START}.nc")
spareice = collect_spareice()
#spareice = xr.open_dataset(f"SPARE-ICE_{START}.nc")

print(spareice, cloudsat)

if False:
    #fig, ax = plt.subplots(figsize=(10, 8))
    #sdata = xr.open_dataset("spareice_2007.nc")
    #lat_bins = np.arange(-90, 85, 5)
    #ax.hist(sdata.lat.values, bins=lat_bins)
    #ax.set_xlabel("latitude [$^\circ$]")
    #ax.set_ylabel("number")
    #ax.set_title("2007 SPARE-ICE distribution")
    #fig.tight_layout()
    #fig.savefig("spareice_lats_2007.png")
    #exit()

    fig, ax = plt.subplots(figsize=(10, 8))
    cdata = xr.open_dataset("2C-ICE_2007.nc")
    ax.hist(cdata.lat.values, bins=lat_bins)
    ax.set_xlabel("latitude [$^\circ$]")
    ax.set_ylabel("number")
    ax.set_title("2007 2C-ICE distribution")
    fig.tight_layout()
    fig.savefig("2c-ice_lats_2007.png")


fig, ax = plt.subplots(figsize=(10, 8))
print("Bin SPARE-ICE...")
plot_zonal_mean(ax, spareice.lat, spareice.iwp, "SPARE-ICE (typhon)")
print("Bin 2C-ICE...")
plot_zonal_mean(ax, cloudsat.lat, cloudsat.ice_water_path, "2C-ICE")
ax.legend()
ax.set_xlabel("latitude [$^\circ$]")
ax.set_ylabel("IWP [$g/m^2$]")
ax.set_title(f"{START} Zonal Mean Comparison")
fig.tight_layout()
fig.savefig(f"zonal_mean_{START}.png")

# fig, ax = plt.subplots(figsize=(10, 8))
# print("Bin SPARE-ICE...")
# plot_zonal_mean(ax, spareice.lat, np.log10(spareice.iwp), "SPARE-ICE (typhon)")
# print("Bin 2C-ICE...")
# iwp = cloudsat.ice_water_path
# plot_zonal_mean(ax, cloudsat.lat[iwp != 0], np.log10(iwp[iwp != 0]), "2C-ICE")
# ax.legend()
# ax.set_xlabel("latitude [$^\circ$]")
# ax.set_ylabel("log10 IWP [$g/m^2$]")
# ax.set_title(f"{START} Zonal Mean Comparison")
# fig.tight_layout()
# fig.savefig(f"zonal_mean_log10_{START}.png")
