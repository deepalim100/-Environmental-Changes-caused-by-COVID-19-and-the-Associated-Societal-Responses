# Importing libraries
import rasterio
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob
import geopandas as gpd
from rasterio.features import shapes
from rasterio.mask import mask
import gdal
import osgeo
from shapely.geometry import box
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio.mask import mask
import fiona


class Urban_growth_detect:
    def __init__(self, input_dir, output_dir, aoi):
        """Here, Input dir : Contains the list of input images,
                Output_dir : Contains the list of output images
                AOI : Region of Interest."""
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._aoi = aoi

    """Image clipping function"""

    def clip_img(self, shapefile, Parent_img_path, clip_img_path):
        # Read Shape file
        with fiona.open(shapefile, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
            # read imagery file
        with rasterio.open(Parent_img_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta
        # Save clipped imagery
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        with rasterio.open(clip_img_path, "w", **out_meta) as dest:
            dest.write(out_image)
        return clip_img_path

    """Clip images"""
    def clip(self):
        # composite-pre clip
        shapefile = self._aoi
        img_path = self._input_dir[0]
        clip_img_comp_pre = self._output_dir[0]
        self.clip_img(shapefile, img_path, clip_img_comp_pre)
        # composite-post clip
        img_path = self._input_dir[1]
        clip_img_comp_post = self._output_dir[1]
        self.clip_img(shapefile, img_path, clip_img_comp_post)
        # nir-pre clip
        img_path = self._input_dir[2]
        clip_img_nir_pre = self._output_dir[2]
        self.clip_img(shapefile, img_path, clip_img_nir_pre)
        # swir pre clip
        img_path = self._input_dir[3]
        clip_img_swir_pre = self._output_dir[3]
        self.clip_img(shapefile, img_path, clip_img_swir_pre)
        # red pre clip
        img_path = self._input_dir[4]
        clip_img_red_pre = self._output_dir[4]
        self.clip_img(shapefile, img_path, clip_img_red_pre)
        # nir post clip
        img_path = self._input_dir[5]
        clip_img_nir_post = self._output_dir[5]
        self.clip_img(shapefile, img_path, clip_img_nir_post)
        # swir post clip
        img_path = self._input_dir[6]
        clip_img_swir_post = self._output_dir[6]
        self.clip_img(shapefile, img_path, clip_img_swir_post)
        # red post clip
        img_path = self._input_dir[7]
        clip_img_red_post = self._output_dir[7]
        self.clip_img(shapefile, img_path, clip_img_red_post)

    """save raster file"""
    def geo_reference(self, parent_img_path, save_img_path, save_array):
        # reading parent image path
        parent_img = rasterio.open(parent_img_path)
        parent_meta = parent_img.meta
        # meta update
        parent_meta.update({"count": 1})
        parent_meta.update({"dtype": 'float32'})
        parent_meta.update({'driver': 'GTiff'})
        # save image path
        array = save_array.astype('float32')
        with rasterio.open(save_img_path, 'w', **parent_meta) as dest:
            dest.write(array, 1)
        return save_img_path

    """Urban change detect function"""
    def NDBI_calculate(self):
        # pre
        pre_nir = self._output_dir[2]
        pre_swir = self._output_dir[3]
        nir_pre = rasterio.open(pre_nir)
        nir_pre_data = nir_pre.read(1)
        swir_pre = rasterio.open(pre_swir)
        swir_pre_data = swir_pre.read(1)
        # NBR
        np.seterr(divide='ignore', invalid='ignore')
        NDBI_pre = (swir_pre_data.astype('float32') - nir_pre_data.astype('float32')) / (
                swir_pre_data.astype('float32') + nir_pre_data.astype('float32'))
        # save image
        inp_path = self._output_dir[1]
        out_path = self._output_dir[8]
        self.geo_reference(inp_path, out_path, NDBI_pre)
        # post
        post_nir = self._output_dir[5]
        post_swir = self._output_dir[6]
        nir_post = rasterio.open(post_nir)
        nir_post_data = nir_post.read(1)
        swir_post = rasterio.open(post_swir)
        swir_post_data = swir_post.read(1)
        # NBR
        np.seterr(divide='ignore', invalid='ignore')
        NDBI_post = (swir_post_data.astype('float32') - nir_post_data.astype('float32')) / (
                swir_post_data.astype('float32') + nir_post_data.astype('float32'))
        # save image
        inp_path = self._output_dir[1]
        out_path = self._output_dir[9]
        self.geo_reference(inp_path, out_path, NDBI_post)
        # NDVI pre
        filename_red = self._output_dir[4]
        filename_nir = self._output_dir[2]
        with rasterio.open(filename_red) as src_red:
            band_red = src_red.read(1)
        with rasterio.open(filename_nir) as src_nir:
            band_nir = src_nir.read(1)
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_pre = (band_nir.astype(float) - band_red.astype(float)) / (band_nir.astype(float) + band_red.astype(float))
        # save image
        inp_path = self._output_dir[1]
        out_path = self._output_dir[10]
        self.geo_reference(inp_path, out_path, ndvi_pre)
        # NDVI post
        filename_red = self._output_dir[7]
        filename_nir = self._output_dir[5]
        with rasterio.open(filename_red) as src_red:
            band_red = src_red.read(1)
        with rasterio.open(filename_nir) as src_nir:
            band_nir = src_nir.read(1)
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_post = (band_nir.astype(float) - band_red.astype(float)) / (band_nir.astype(float) + band_red.astype(float))
        # save image
        inp_path = self._output_dir[1]
        out_path = self._output_dir[11]
        self.geo_reference(inp_path, out_path, ndvi_post)



if __name__ == "__main__":
    input_dir = [r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\res\2020\landsat_rgb.tiff",
                 r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\res\2021\landsat_rgb_2021.tiff",
                 r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\2020\LC08_L2SP_122045_20201202_20210312_02_T1_SR_B5.TIF",
                 r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\2020\LC08_L2SP_122045_20201202_20210312_02_T1_SR_B6.TIF",
                 r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\2020\LC08_L2SP_122045_20201202_20210312_02_T1_SR_B4.TIF",
                 r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\2021\LC08_L2SP_122045_20210220_20210303_02_T1_SR_B5.TIF",
                 r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\2021\LC08_L2SP_122045_20210220_20210303_02_T1_SR_B6.TIF",
                 r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\2021\LC08_L2SP_122045_20210220_20210303_02_T1_SR_B4.TIF"
                 ]

    output_dir = [r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\composite_2020_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\composite_2021_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\nir_2020_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\swir_2020_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\red_2020_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\nir_2021_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\swir_2021_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\red_2021_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\ndbi_2020_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\ndbi_2021_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\ndvi_2020_clip.tif",
                  r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\landsat\script_res\ndvi_2021_clip.tif"
                  ]
    shapefile = r"C:\Deepali_pro\Covid_19_impact\urban_growth\result\AOI_2_based_res\aoi\aoi.shp"
    Urban_growth_detect(input_dir=input_dir, output_dir=output_dir, aoi=shapefile).clip()
    Urban_growth_detect(input_dir=input_dir, output_dir=output_dir, aoi=shapefile).NDBI_calculate()
    print("Process Finished !!!")
