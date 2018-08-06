#!/usr/bin/env python3

"""This script produces the training data for SPARE-ICE

This script performs three stages to create the training dataset for SPARE-ICE.
It requires the original 2C-ICE, MHS and AVHRR GAC files.

1) Collocate MHS and 2C-ICE (*T_max = 5 min* and *S_max = 7.5 km*) and
store the collocations.
2) Collocate the collocated MHS-2C-ICE dataset with AVHRR (*T_max = 5 min* and
*S_max = 7.5 km*) and store these collocations as well.
3) Collect the collocations, apply the homogeneity filters and store the
relevant fields to a NetCDF file.

"""
import os
os.environ["OMP_NUM_THREADS"] = "2"  # noqa

from typhon.collocations import Collocations
from typhon.files import FileSet
from typhon.files import AVHRR_GAC_HDF, CloudSat, MHS_HDF, NetCDF4
import xarray as xr

STAGE_ONE = False
STAGE_TWO = False
STAGE_THREE = True
START_TIME = "2007"
END_TIME = "March 2010"
PROCESSES = 12

TRAINING_FILE = "spareice_training_data.nc"

# Define a fileset with the files from MHS / NOAA18:
mhs = FileSet(
    name="MHS",
    path="/work/um0878/data/amsub_mhs_l1c_hdf/AAPP7_13/noaa18"
         "_mhs_{year}/{month}/{day}/*NSS.MHSX.NN.*."
         "S{hour}{minute}.E{end_hour}{end_minute}.*.h5",
    handler=MHS_HDF(),
    # Load only the fields that we need:
    read_args={
        "fields": [
            "Data/btemps",
            "Geolocation/Satellite_azimuth_angle",
            "Geolocation/Satellite_zenith_angle",
            "Geolocation/Solar_azimuth_angle",
            "Geolocation/Solar_zenith_angle",
        ]
    },
)

# Define a fileset with files from CloudSat / 2C-ICE:
cloudsat = FileSet(
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
    }
)

# Define where the collocations should be stored:
mhs_2cice = Collocations(
    name="MHS_2C-ICE",
    path="/work/um0878/user_data/jmrziglod/collocations/MHS_2C-ICE/"
         "Mrziglod2018/{year}/{month}/{day}/{hour}{minute}{second}-"
         "{end_hour}{end_minute}{end_second}.nc",
    # We want to collapse everything to MHS pixels:
    reference="MHS"
)

# STAGE ONE - Collocate MHS with 2C-ICE:
if STAGE_ONE:
    # Search for collocations between MHS and 2C-ICE:
    mhs_2cice.search(
        [mhs, cloudsat], start=START_TIME, end=END_TIME, processes=PROCESSES,
        max_interval="10 min", max_distance="7.5 km", verbose=2,
        skip_file_errors=True,
    )


# Define a fileset with the files from AVHRR / NOAA18:
avhrr = FileSet(
    name="AVHRR",
    path="/work/um0878/user_data/jmrziglod/avhrr_gac_hdf5/noaa18_gac_"
         "{year}/{month}/{day}/NSS.*."
         "S{hour}{minute}.E{end_hour}{end_minute}.*.h5",
    handler=AVHRR_GAC_HDF(),
    # Load only the fields that we need:
    read_args={"fields": [
            "Data/btemps",
            "Geolocation/Relative_azimuth_angle",
        ]
    },
)

# Define where the collocations should be stored:
mhs_2cice_avhrr = Collocations(
    path="/work/um0878/user_data/jmrziglod/collocations/MHS_2C-ICE_AVHRR/"
         "Mrziglod2018/{year}/{month}/{day}/{hour}{minute}{second}-"
         "{end_hour}{end_minute}{end_second}.nc",
    # We want to collapse everything to MHS pixels:
    reference="MHS_2C-ICE",
)

# STAGE TWO - Collocate MHS-2C-ICE with AVHRR:
if STAGE_TWO:
    # Search for collocations between MHS and 2C-ICE:
    mhs_2cice_avhrr.search(
        [avhrr, mhs_2cice], start=START_TIME, end=END_TIME,
        processes=PROCESSES, max_interval="30s", max_distance="7.5 km",
        verbose=2, skip_file_errors=True,
    )

# STAGE THREE - Collect the collocations, apply the filters and store the data:
if STAGE_THREE:
    print("Create the training dataset")
    raw_data = xr.concat(
        mhs_2cice_avhrr.collect(START_TIME, END_TIME, max_workers=20),
        dim="collocation"
    )

    print(raw_data)

    # We apply a homogeneity filter. Why? We collapsed multiple 2C-ICE pixels
    # to one MHS pixel and assume that they are representative for the whole
    # MHS pixel. But honestly, we do not know it since they cover maximal 6.5%
    # of the MHS pixel. The only thing that we can do, is assuring that all of
    # the 2C-ICE pixels saw roughly a very similar atmosphere.
    good = raw_data["MHS_2C-ICE/2C-ICE/ice_water_path_number"] > 10
    # Filter out the IWPs that are 0 g/m^2.
    good &= raw_data["MHS_2C-ICE/2C-ICE/ice_water_path_std"] \
        / raw_data["MHS_2C-ICE/2C-ICE/ice_water_path_mean"] < 0.5
    data = raw_data.sel(collocation=good)

    print(data)

    NetCDF4().write(data, TRAINING_FILE)
    print(f"Store to {TRAINING_FILE}")
