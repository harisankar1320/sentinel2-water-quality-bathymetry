import zipfile
import rasterio
import os
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import glob
import warnings
import gc


PROCESSED_DATA_PATH = r"F:\data_water\processed"
RAW_EXTRACTED       = r"F:\data_water\raw_extracted"
indices_path = os.path.join(PROCESSED_DATA_PATH, "indices")
os.makedirs(indices_path, exist_ok=True)
bands_path = os.path.join(PROCESSED_DATA_PATH, "bands")
os.makedirs(bands_path, exist_ok=True)
rgb_path = os.path.join(PROCESSED_DATA_PATH, "rgb")
os.makedirs(rgb_path, exist_ok=True)


def find_band_files(scene):        # finds all jp2 files in RAW EXTRACTED (extracted zip files)

    """
    Find all band file paths for one .SAFE scene.
    
    Parameters:
        scene_path: str, full path to one .SAFE folder
    
    Returns:
        dict with paths for each band
    """

    b02 = glob.glob(os.path.join(scene, '**/*_B02_10m.jp2'), recursive=True)
    b03 = glob.glob(os.path.join(scene, '**/*_B03_10m.jp2'), recursive=True)
    b04 = glob.glob(os.path.join(scene, '**/*_B04_10m.jp2'), recursive=True)
    b08 = glob.glob(os.path.join(scene, '**/*_B08_10m.jp2'), recursive=True)
    scl = glob.glob(os.path.join(scene, '**/*_SCL_20m.jp2'), recursive=True)

    # Check all files found
    if not all([b02, b03, b04, b08, scl]):
        missing = [name for name, f in 
                   zip(['b02','b03','b04','b08','scl'], [b02,b03,b04,b08,scl]) 
                   if not f]
        print(f"Missing bands {missing} in {scene} — skipping")
        return None

    return {
        'b02': b02[0],
        'b03': b03[0],
        'b04': b04[0],
        'b08': b08[0],
        'scl': scl[0]
    }



def read_and_mask_band(b02, b03, b04, b08, scl, glint_factor = 0.5): # read + glint correct
    print("Starting read and mask")
    with rasterio.open(scl) as src:
        scl = src.read(1, out_shape=(10980, 10980), resampling=Resampling.nearest).astype('float32')
        # profile = src.profile
        # print("SCL shape:", scl.shape)
        # print(profile)
        # print("** scl done **")
    valid_masks = (scl == 6) # 6 is water , so keeps only water pixels
      

    # print("Starting b08")
    with rasterio.open(b08) as src:
        b08 = src.read(1).astype('float32')

        ## REFLECTANCE (default unit) is: DN = 10000 * REFLECTANCE, for physical algorithms (chlorophyll, turbidity, bathymetry) convert to reflectance first, Coefficients assume 0-1 range

        b08_reflectance = b08 / 10000
        b08_masked = np.where(valid_masks, b08_reflectance, np.nan)
        # print("** b08 done **\n")

    # print("** Starting b02 **")

    with rasterio.open(b02) as src:
        b02 = src.read(1).astype('float32')
        profile = src.profile
        # print("b02 shape: ", b02.shape)
        # print(profile)
        # print("b02 min:", np.nanmin(b02))
        # print("b02 max:", np.nanmax(b02))

        b02_reflectance = b02 / 10000
        b02_masked = np.where(valid_masks, b02_reflectance, np.nan)

        # NIR over deep water should be ~0
        # Any NIR signal = glint
        # Subtract from visible bands # a glint factor of 0.5 is used after trying subtracting nir since a lot of pixels was going negative and then nan
        b02_corrected = b02_masked - (glint_factor * b08_masked)    
        b02_corrected = np.where(b02_corrected < 0, np.nan, b02_corrected)  
        # print("b02_corrected min:", np.nanmin(b02_corrected))
        # print("b02_corrected max:", np.nanmax(b02_corrected))

    # After creating b02_corrected, delete raw arrays
    del b02, b02_reflectance, b02_masked
    gc.collect()
        ##Total valid pixels
    #     print("Total valid pixels for b02: ", np.sum(~np.isnan(b02_corrected)))
    #     print("Total pixels ob b02: ", b02_corrected.size)
    #     print("water percentage:", round(np.sum(~np.isnan(b02_corrected)) / b02_corrected.size * 100, 2), "%")
 
    #     print("** b02 done **\n")

    # print("** Starting b03 **")
    with rasterio.open(b03) as src:
        b03 = src.read(1).astype('float32')
        profile = src.profile
        # print("b03 shape: ",b03.shape)
        # print(profile)
        # print("b03 min:", b03.min())
        # print("b03 max:", b03.max())

        b03_reflectance = b03 / 10000
        b03_masked = np.where(valid_masks, b03_reflectance, np.nan)

        #glint removal
        b03_corrected = b03_masked - (glint_factor * b08_masked  )   
        b03_corrected = np.where(b03_corrected < 0, np.nan, b03_corrected)  
        # print("b03_corrected min:", np.nanmin(b03_corrected))
        # print("b03_corrected max:", np.nanmax(b03_corrected))
    # After creating b02_corrected, delete raw arrays
    del b03, b03_reflectance, b03_masked
    gc.collect()
         ##Total valid pixels
    #     print("Total valid pixels for b03: ", np.sum(~np.isnan(b03_corrected)))
    #     print("Total pixels ob b03: ", b03_corrected.size)
    #     print("water percentage:", round(np.sum(~np.isnan(b03_corrected)) / b03_corrected.size * 100, 2), "%")


    #     print("** b03 done **\n")

      
    # print("** Starting b04 **")
    with rasterio.open(b04) as src:
        b04 = src.read(1).astype('float32')
        out_profile = src.profile        # profile of b04 is chosen since its one of 10 m resolution and the out meta data should be 10m res
        # print("b04 shape:",b04.shape)
        # # print(profile)
        # print("b04 min:", b04.min())
        # print("b04 max:", b04.max())

        b04_reflectance = b04 / 10000
        b04_masked = np.where(valid_masks, b04_reflectance, np.nan)

        #glint removal
        b04_corrected = b04_masked - (glint_factor * b08_masked )    
        b04_corrected = np.where(b04_corrected < 0, np.nan, b04_corrected)  
        # print("b04_corrected min:", np.nanmin(b04_corrected))
        # print("b04_corrected max:", np.nanmax(b04_corrected))
    # After creating b02_corrected, delete raw arrays
    del b04, b04_reflectance, b04_masked
    gc.collect()
        ##Total valid pixels
        # print("Total valid pixels for b04: ", np.sum(~np.isnan(b04_corrected)))
        # print("Total pixels ob b04: ", b04_corrected.size)
        # print("water percentage:", round(np.sum(~np.isnan(b04_corrected)) / b04_corrected.size * 100, 2), "%")

        # print("** b04 done **\n")

    combined_mask = (
        ~np.isnan(b02_corrected) &
        ~np.isnan(b03_corrected) &
        ~np.isnan(b04_corrected)
      )

    b08_final = np.where(combined_mask, b08_masked, np.nan)
    b02_final = np.where(combined_mask, b02_corrected, np.nan)
    b03_final = np.where(combined_mask, b03_corrected, np.nan)
    b04_final = np.where(combined_mask, b04_corrected, np.nan)

    # print("Final consistent pixels: ", combined_mask.sum())
    # print("Percentage:", round(combined_mask.sum() / b02_corrected.size * 100, 2), "%")

    print("Finished read and mask")

    return b02_final, b03_final, b04_final,b08_final, out_profile



 # INDICES Calculation for LSTM (NDWI, Chlorophyll and turbidity)
def calculate_indices(b02, b03, b04, b08):        # returns NDWI, turbidity, chl

    print("Starting calculate_indices")

    #Normalise water index
    NDWI = (b03 - b08) / (b03 + b08)

    # Based on Dogliotti et al. 2015, Turbidity
    turbidity = 1.73 * b04 / (1 - b04 / 0.17)
    # mask invalid turbidity pixels
    turbidity = np.where(b04 >= 0.17, np.nan, turbidity)
    turbidity = np.where(turbidity < 0, np.nan, turbidity)
    turbidity = np.where(turbidity > 100, np.nan, turbidity)

    # Based on OC2 algorithm (O'Reilly et al.)
    ratio = np.log10(b02 / b03)
    ratio = np.where((ratio < -0.2) | (ratio > 0.4), np.nan, ratio)
    chlorophyll = 10 ** (0.2424 - 2.7423 * ratio + 1.8017 * ratio**2)
    chlorophyll = np.where(chlorophyll > 100, np.nan, chlorophyll)  # realistic max ~100 mg/m³

    # print("NDWI min/max:", np.nanmin(NDWI), np.nanmax(NDWI))
    # print("Turbidity min/max:", np.nanmin(turbidity), np.nanmax(turbidity))
    # print("Chlorophyll min/max:", np.nanmin(chlorophyll), np.nanmax(chlorophyll))
    # print("ratio min/max:", np.nanmin(ratio), np.nanmax(ratio))
    # print("b02/b03 ratio min/max:", np.nanmin(b02_final/b03_final), np.nanmax(b02_final/b03_final))
    print("Finished calculate_indices")

    return NDWI, turbidity, chlorophyll

def save_tif(bands, indices, out_profile, indices_path, bands_path, date):

    print("Starting save_tif")

    #SAVE BANDS INDICES AND RGBS
    out_profile['dtype'] = 'float32'
    out_profile['count'] = 3
    out_profile['nodata'] = np.nan
    out_profile['driver'] = 'GTiff'

    with rasterio.open(indices_path, 'w', **out_profile) as dst:
        dst.write(indices)
        print(f"  Indices for {date} Completed")

    with rasterio.open(bands_path, 'w', **out_profile) as dst:
        dst.write(bands)
        print(f"  Bands for {date} Completed")

    print("Finished save_tif")


def read_and_save_rgb(bands, path, date):
    """Read raw reflectance for RGB visualization only."""

    print("Starting read_and_save_rgb")

    with rasterio.open(bands['b04']) as src:
        b04 = src.read(1).astype('float32') / 10000
    with rasterio.open(bands['b03']) as src:
        b03 = src.read(1).astype('float32') / 10000
    with rasterio.open(bands['b02']) as src:
        b02 = src.read(1).astype('float32') / 10000

    # Downsample to 1/4 resolution for visualization
    # 10980 → 2745 pixels — much more manageable
    b04 = b04[::4, ::4]
    b03 = b03[::4, ::4]
    b02 = b02[::4, ::4]

    rgb = np.dstack([b04, b03, b02])
    for i in range(3):
        p2  = np.nanpercentile(rgb[:,:,i], 2)
        p98 = np.nanpercentile(rgb[:,:,i], 98)
        rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2), 0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(f'True Color RGB - {date}')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  RGB for {date} saved")

    print("Finished read_and_save_rgb")


for scenes in os.listdir(RAW_EXTRACTED):
    if scenes.endswith('.SAFE'):
        date = scenes[11:19]

        out_indices = os.path.join(indices_path, f"indices {date}.tif")
        out_bands =os.path.join(bands_path, f"Bands {date}.tif")
        out_rgb = os.path.join(rgb_path, f"rgb {date}.png")

        if os.path.exists(out_indices) and os.path.exists(out_bands) and os.path.exists(out_rgb):
            print(f"File {scenes} already processed")
            continue
        
        #FIND BANDS AND FILES FOR EACH SCENES    
        print(f"Starting processing {date} .. .tif")
        scene_path = os.path.join(RAW_EXTRACTED, scenes)
        bands = find_band_files(scene_path)
        if bands is None:
            continue
        #PROCESS
        b02_final, b03_final, b04_final, b08_final, out_profile = read_and_mask_band(
            bands['b02'], bands['b03'], bands['b04'], bands['b08'], bands['scl']
            )

        #INDICES
        NDWI, turbidity, chlorophyll =  calculate_indices(b02_final, b03_final, b04_final, b08_final)

        #SAVE BANDS INDICES AND RGBS
        stack_bands = np.stack([b02_final, b03_final, b04_final])
        stack_indices = np.stack([NDWI, turbidity, chlorophyll])
        save_tif(stack_bands, stack_indices, out_profile, out_indices, out_bands, date)
        read_and_save_rgb(bands, out_rgb, date) 
        print(f"Processing completed {date} .. .tif")   
        # Free memory after each scene
        del b02_final, b03_final, b04_final, b08_final
        del stack_bands, stack_indices
        del NDWI, turbidity, chlorophyll
        gc.collect()
        print(f"Memory freed after {date}") 

print("ALLES GUT")   









