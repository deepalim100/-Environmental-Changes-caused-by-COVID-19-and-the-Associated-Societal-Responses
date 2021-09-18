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


class Deforest_detect:
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
        # red pre clip
        img_path = self._input_dir[3]
        clip_img_red_pre = self._output_dir[3]
        self.clip_img(shapefile, img_path, clip_img_red_pre)
        # nir post clip
        img_path = self._input_dir[4]
        clip_img_nir_post = self._output_dir[4]
        self.clip_img(shapefile, img_path, clip_img_nir_post)
        # red post clip
        img_path = self._input_dir[5]
        clip_img_red_post = self._output_dir[5]
        self.clip_img(shapefile, img_path, clip_img_red_post)


    # NDVI of images
    def generate_ndvi_image(self,inp_path, out_path):
        filename_red = inp_path[0]
        filename_nir = inp_path[1]

        with rasterio.open(filename_red) as src_red:
            band_red = src_red.read(1)

        with rasterio.open(filename_nir) as src_nir:
            band_nir = src_nir.read(1)

        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir.astype(float) + band_red.astype(float))

        class MidpointNormalize(colors.Normalize):
            """
            Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
            e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
            Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
            """

            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [-1, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

        min = np.nanmin(ndvi)
        max = np.nanmax(ndvi)
        mid = 0.1
        norm = MidpointNormalize(midpoint=mid, vmin=min, vmax=max)
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        cmap = plt.cm.RdYlGn
        cax = ax.imshow(ndvi, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid, vmin=min, vmax=max))
        ax.axis('off')
        ax.set_title('Normalized Difference Vegetation Index', fontsize=18, fontweight='bold')
        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)

        fig.savefig(out_path[0], dpi=200, bbox_inches='tight',
                    pad_inches=0.7)

        ndvi_img_path = out_path[1]
        band2_geo = src_nir.meta
        # band2_geo.update({"count": 6})
        ndvi_1 = ndvi[np.newaxis, :, :]
        band2_geo.update({"count": 1})
        band2_geo.update({"dtype": 'float64'})
        band2_geo.update({'driver': 'GTiff'})

        with rasterio.open(ndvi_img_path, 'w', **band2_geo) as dest:
            dest.write(ndvi_1)

        return ndvi_img_path

    def change_detect(self, inp1, inp2, out):
        input_1 = rasterio.open(inp1).read(1)
        input_2 = rasterio.open(inp2).read(1)
        ndvi_final = input_1 - input_2
        ndvi_norm = (255 * ndvi_final / np.max(ndvi_final)).astype(np.uint8)
        vmin, vmax = np.nanpercentile(ndvi_norm, (1, 99))  # 1-99% contrast stretch
        # Thresholding for detecting the change
        import itertools
        thresh = ndvi_final
        k = float(255)
        for i, j in itertools.product(range(thresh.shape[0]), range(thresh.shape[1])):
            if thresh[i][j] >= -0.10 and thresh[i][j] <= -0.05:
                thresh[i][j] = 1.0
            else:
                thresh[i][j] = 0.0
        plt.figure(figsize=(12, 12))
        img_plt = plt.imshow(thresh, cmap='gray')
        plt.title('Deforestation')
        plt.colorbar()
        # save image
        # save ndvi change file
        with rasterio.open(inp1) as src:
            out_image = src.read(1)
            out_meta = src.meta
        # Save clipped imagery
        out_meta.update({"driver": "GTiff", "count": 1})
        with rasterio.open(out, "w", **out_meta) as dest:
            dest.write(thresh.astype('float64'), 1)



if __name__ == "__main__":
    input_dir = [r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\2019\L1C_T47NMD_A020793_20190616T035452rgb.tiff",
                 r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\2021\L1C_T47NMD_A022538_20210630T035418rgb.tiff",
                 r"C:\Deepali_pro\Covid_19_impact\Deforestation\L1C_T47NMD_A020793_20190616T035452\S2A_MSIL1C_20190616T033541_N0207_R061_T47NMD_20190616T071029.SAFE\GRANULE\L1C_T47NMD_A020793_20190616T035452\IMG_DATA\T47NMD_20190616T033541_B08.jp2",
                 r"C:\Deepali_pro\Covid_19_impact\Deforestation\L1C_T47NMD_A020793_20190616T035452\S2A_MSIL1C_20190616T033541_N0207_R061_T47NMD_20190616T071029.SAFE\GRANULE\L1C_T47NMD_A020793_20190616T035452\IMG_DATA\T47NMD_20190616T033541_B04.jp2",
                 r"C:\Deepali_pro\Covid_19_impact\Deforestation\L1C_T47NMD_A022538_20210630T035418\S2B_MSIL1C_20210630T033539_N0300_R061_T47NMD_20210630T071530.SAFE\GRANULE\L1C_T47NMD_A022538_20210630T035418\IMG_DATA\T47NMD_20210630T033539_B08.jp2",
                 r"C:\Deepali_pro\Covid_19_impact\Deforestation\L1C_T47NMD_A022538_20210630T035418\S2B_MSIL1C_20210630T033539_N0300_R061_T47NMD_20210630T071530.SAFE\GRANULE\L1C_T47NMD_A022538_20210630T035418\IMG_DATA\T47NMD_20210630T033539_B04.jp2"
                 ]
    output_dir = [r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\composite_2019.tif",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\composite_2021.tif",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\nir_2019.tif",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\red_2019.tif",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\nir_2021.tif",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\red_2021.tif"
                  ]
    shapefile = r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\AOI\clip-1\Tanjung_lankat.shp"
    inp_path_1 = [r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\red_2019.tif",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\nir_2019.tif"]
    inp_path_2 = [r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\red_2021.tif",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\nir_2021.tif"]
    out_path_1 = [r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\ndvi_2019.png",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\ndvi_2019.tif"]
    out_path_2 = [r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\ndvi_2021.png",
                  r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\ndvi_2021.tif"]
    Deforest_detect(input_dir=input_dir, output_dir=output_dir, aoi=shapefile).clip()
    #NDVI - 2019
    ndvi_2019_path = Deforest_detect(input_dir=input_dir, output_dir=output_dir, aoi=shapefile).generate_ndvi_image(inp_path=inp_path_1,out_path=out_path_1)
    #NDVI - 2021
    ndvi_2021_path = Deforest_detect(input_dir=input_dir, output_dir=output_dir, aoi=shapefile).generate_ndvi_image(inp_path=inp_path_2,out_path=out_path_2)
    #Change detect
    out_path = r"C:\Deepali_pro\Covid_19_impact\Deforestation\result\result_script\Change.tif"
    Deforest_detect(input_dir=input_dir, output_dir=output_dir, aoi = shapefile).change_detect(ndvi_2021_path,ndvi_2019_path, out_path)
    print("Process Completed!!")