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
from typhon.retrieval import SPAREICE
from typhon.files import AVHRR_GAC_HDF, FileSet, MHS_HDF

START_TIME = "20 Jun 2013"
END_TIME = "21 Jun 2013"
PROCESSES = 1
SAVE_COLLOCATIONS = False

version = "typhon"

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
    path="/work/um0878/user_data/jmrziglod/spareice/{version}/noaa18/"
         "{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}-"
         "{end_hour}{end_minute}{end_second}.nc",
)
spareice_files.set_placeholders(version=version)

spareice = SPAREICE(
     verbose=2, sea_mask_file="data/land_water_mask_5min.png",
     elevation_file="data/surface_elevation_1deg.nc",
)

# Either we search once for the collocations, save them to disk and retrieve
# SPARE-ICE on them or we search for the collocations each time when retrieving
# SPARE-ICE. The first approach is obviously faster.

if SAVE_COLLOCATIONS:
    collocations = Collocations(
        path="/work/um0878/user_data/jmrziglod/collocations/MHS_AVHRR/noaa18/"
             "{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}-"
             "{end_hour}{end_minute}{end_second}.nc",
        reference="MHS",
    )
    collocations.search(
        [avhrr_files, mhs_files], start=START_TIME, end=END_TIME,
        processes=PROCESSES, max_interval="30s", max_distance="7.5 km",
    )

    # Use the already collocated data:
    input_filesets = collocations
else:
    # Search again after collocations:
    input_filesets = [avhrr_files, mhs_files]

# Retrieve SPARE-ICE:
spareice.retrieve_from_collocations(
    inputs=input_filesets, output=spareice_files,
    start=START_TIME, end=END_TIME, processes=PROCESSES,
)


