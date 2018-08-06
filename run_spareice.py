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

from typhon.retrieval import SPAREICE
from typhon.files import AVHRR_GAC_HDF, FileSet, MHS_HDF

START_TIME = "2007-06-19 00:41:00"
END_TIME = "2007-07-17 09:46:00"
PROCESSES = 10


# Define a fileset with the files from MHS / NOAA18:
mhs_files = FileSet(
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

# Define a fileset with the files from AVHRR / NOAA18:
avhrr_files = FileSet(
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

spareice_files = FileSet(
    name="SPAREICE",
    path="/work/um0878/user_data/jmrziglod/spareice/noaa18/"
         "{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}-"
         "{end_hour}{end_minute}{end_second}.nc",
)

spareice = SPAREICE(verbose=2)
spareice.retrieve_from_filesets(
    mhs=mhs_files, avhrr=avhrr_files, output=spareice_files,
    start=START_TIME, end=END_TIME, processes=PROCESSES,
    sea_mask_file="land_water_mask_5min.png",
)
