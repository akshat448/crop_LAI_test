import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_tif_files(image_path):
    """
    Get all the .tif files in the image folder.

    Parameters
    ----------
    image_path: pathlib Path
        Directory to search for tif files
    Returns:
        A list of .tif filenames
    """
    files = []
    for dir_file in image_path.iterdir():
        if str(dir_file).endswith('tif'):
            # strip out the directory so its just the filename
            files.append(str(dir_file.parts[-1]))
    print(len(files))
    return files


def load_clean_yield_data(yield_data_filepath):
    """
    Cleans the yield data by making sure any Nan values in the columns we care about
    are removed
    """
    important_columns = ['Year', 'State Name', 'Dist Name', 'RICE YIELD (Kg per ha)']
    yield_data = pd.read_csv(yield_data_filepath,dtype=str).dropna(subset=important_columns, how='any')
    yield_data = yield_data[(yield_data['RICE YIELD (Kg per ha)'] != '-1') & (yield_data['RICE YIELD (Kg per ha)'] != '0')]
    return yield_data

# def load_clean_yield_data(yield_data_filepath):
#     """
#     Cleans the yield data by making sure any NaN values in the columns we care about
#     are removed and any -1 or 0 values in the yield columns are removed.
#     Additionally, subtracts the mean from the 'RICE YIELD (Kg per ha)' column.
#     """
#     important_columns = ['Year', 'State Name', 'Dist Name', 'RICE YIELD (Kg per ha)', 'WHEAT YIELD (Kg per ha)']
#     yield_data = pd.read_csv(yield_data_filepath, dtype=str).dropna(subset=important_columns, how='any')
    
#     # Remove rows with -1 or 0 values in the yield columns
#     yield_data = yield_data[(yield_data['RICE YIELD (Kg per ha)'] != '-1') & 
#                             (yield_data['RICE YIELD (Kg per ha)'] != '0')]
    
#     # Convert 'RICE YIELD (Kg per ha)' to numeric and subtract the mean
#     yield_data['RICE YIELD (Kg per ha)'] = pd.to_numeric(yield_data['RICE YIELD (Kg per ha)'])
#     rice_yield_mean = yield_data['RICE YIELD (Kg per ha)'].mean()
    
#     yield_data['RICE YIELD (Kg per ha)'] = yield_data['RICE YIELD (Kg per ha)'] * 0.01
#     yield_data['RICE YIELD (Kg per ha)'] = yield_data['RICE YIELD (Kg per ha)'] - rice_yield_mean

#     return yield_data


def visualize_modis(data):
    """Visualize a downloaded MODIS file.

    Takes the red, green and blue bands to plot a
    'colour image' of a downloaded tif file.

    Note that this is not a true colour image, since
    this is a complex thing to represent. It is a 'basic
    true colour scheme'
    http://www.hdfeos.org/forums/showthread.php?t=736

    Parameters
    ----------
    data: a rasterio mimic Python file object
    """
    arr_red = data.read(1)
    arr_green = data.read(4)
    arr_blue = data.read(3)

    im = np.dstack((arr_red, arr_green, arr_blue))

    im_norm = im / im.max()

    plt.imshow(im_norm)


def normalize_yield_data(yield_data):
    """
    Normalize the yield data to a 0-1 range.
    """
    min_val = yield_data.min()
    max_val = yield_data.max()
    normalized_data = (yield_data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val

def standardize_yield_data(yield_data):
    """
    Standardize the yield data to have a mean of 0 and a standard deviation of 1.
    """
    mean = yield_data.mean()
    std = yield_data.std()
    standardized_data = (yield_data - mean) / std
    return standardized_data, mean, std

def convert_kg_per_ha_to_bu_per_acre(yield_data):
    """
    Convert yield data from kg/ha to bushels per acre.
    Note: 1 kg/ha = 0.014867 bushels per acre for soybeans.
    """
    conversion_factor = 0.014867
    yield_data_bu_per_acre = yield_data * conversion_factor
    return yield_data_bu_per_acre
