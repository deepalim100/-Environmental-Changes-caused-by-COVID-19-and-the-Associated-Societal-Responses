import rasterio
import numpy as np
import matplotlib.pyplot as plt


class Air_pollution:
    def __init__(self, input_path, output_path, title):
        """Here, Input dir : Contains the list of input images,
                        Output_dir : Contains the list of output images
                        Title : Name for image."""
        self._input_path = input_path
        self._output_path = output_path
        self._title = title

    def process(self):
        data = rasterio.open(self._input_path)
        phase = data.read([1, 2, 3])
        phase = np.moveaxis(phase, 0, -1)
        plt.figure(figsize=(6, 6))
        plt.title(self._title, fontsize=18, fontweight='bold')
        plt.imshow(phase)
        plt.colorbar(orientation='horizontal')
        plt.savefig(self._output_path, dpi=200, bbox_inches='tight',pad_inches=0.7)
        return 0

if __name__=="__main__":
    inp_path = r"C:\Deepali_pro\Covid_19_impact\Air_polution\NO2\2020-04-14-00_00_2020-04-14-23_59_Sentinel-5_NO2_Nitrogen_Dioxide.tiff"
    out_path = r"C:\Deepali_pro\Covid_19_impact\Air_polution\No2_phase1.png"
    title = "NO2 in Phase-1"
    Air_pollution(input_path=inp_path,output_path=out_path,title=title).process()