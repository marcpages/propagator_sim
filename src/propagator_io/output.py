import json
from datetime import datetime
from pathlib import Path

import numpy as np
from pyproj import CRS
from rasterio.transform import Affine

from propagator.models import PropagatorStats
from propagator.utils import write_geotiff


def write_geotiff_output(
    values: np.ndarray,
    dst_trans: Affine,
    dst_crs: CRS,
    output_folder: Path,
    prefix: str,
    c_time: int,
) -> None:
    tiff_file = output_folder / f"{prefix}_{c_time}.tiff"
    # now it returns the RoS in m/h
    write_geotiff(tiff_file, values, dst_trans, dst_crs, values.dtype)


# def __update_isochrones(self, isochrones, values, dst_trans):
#     isochrones[self.c_time] = extract_isochrone(
#         values,
#         dst_trans,
#         thresholds=[0, 0.5, 0.75, 0.9],
#     )

# def __write_isochrones(self, isochrones):
#     isochrone_file = "isochrones_" + str(self.c_time) + ".geojson"
#     isochrone_path = os.path.join(self.settings.output_folder, isochrone_file)
#     save_isochrones(isochrones, isochrone_path, format="geojson")


# def write_all_output():
#     reprj_values, dst_trans = reproject(
#         values,
#         self.__trans,
#         # self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
#         self.__prj.srs,
#         self.dst_crs,
#         trim=True,  # trim is used to auto-clip the tif where there are null value
#     )

#     self.__write_output(
#         reprj_values,
#         dst_trans,
#         active=n_active,
#         area_mean=area_mean,
#         area_50=area_50,
#         area_75=area_75,
#         area_90=area_90,
#     )

#     reprj_values_ros_max, dst_trans_ros_max = reproject(
#         RoS_max,
#         self.__trans,
#         # self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
#         self.__prj.srs,
#         self.dst_crs,
#         trim=True,  # trim is used to auto-clip the tif where there are null value
#     )

#     self.__write_output_RoS_max(
#         reprj_values_ros_max,
#         dst_trans_ros_max,
#         active=n_active,
#         area_mean=area_mean,
#         area_50=area_50,
#         area_75=area_75,
#         area_90=area_90,
#     )

#     reprj_values_ros_mean, dst_trans_ros_mean = reproject(
#         RoS_mean,
#         self.__trans,
#         # self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
#         self.__prj.srs,
#         self.dst_crs,
#         trim=True,  # trim is used to auto-clip the tif where there are null value
#     )

#     self.__write_output_RoS_mean(
#         reprj_values_ros_mean,
#         dst_trans_ros_mean,
#         active=n_active,
#         area_mean=area_mean,
#         area_50=area_50,
#         area_75=area_75,
#         area_90=area_90,
#     )

#     reprj_values_I_max, dst_trans_I_max = reproject(
#         fl_I_max,
#         self.__trans,
#         # self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
#         self.__prj.srs,
#         self.dst_crs,
#         trim=True,  # trim is used to auto-clip the tif where there are null value
#     )

#     self.__write_output_I_max(
#         reprj_values_I_max,
#         dst_trans_I_max,
#         active=n_active,
#         area_mean=area_mean,
#         area_50=area_50,
#         area_75=area_75,
#         area_90=area_90,
#     )

#     reprj_values_I_mean, dst_trans_I_mean = reproject(
#         fl_I_mean,
#         self.__trans,
#         # self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
#         self.__prj.srs,
#         self.dst_crs,
#         trim=True,  # trim is used to auto-clip the tif where there are null value
#     )

#     self.__write_output_I_mean(
#         reprj_values_I_mean,
#         dst_trans_I_mean,
#         active=n_active,
#         area_mean=area_mean,
#         area_50=area_50,
#         area_75=area_75,
#         area_90=area_90,
#     )

#     self.__update_isochrones(isochrones, reprj_values, dst_trans)
#     self.__write_isochrones(isochrones)


# normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
