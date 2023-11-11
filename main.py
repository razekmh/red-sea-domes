# %%
import numpy as np
from skimage import feature, measure
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# %%
parser = argparse.ArgumentParser(
    prog='Red Sea Dome Detection using LoG',
    description='This program detects the domes in the Red Sea depth map using Laplacian of Gaussian (LoG) filter.',
    epilog='Please feel free to open an issue at the repo if you face any problem')

parser.add_argument('-ns', '--min_sigma', type=int, default=0, help='Minimum standard deviation for Gaussian kernel. Default: 0')
parser.add_argument('-xs', '--max_sigma', type=int, default=15, help='Maximum standard deviation for Gaussian kernel. Default: 15')
parser.add_argument('-t', '--threshold', type=float, default=0.01, help='Threshold value to detect blobs. Default: 0.01')
parser.add_argument('-rs', '--radius_scaler', type=int, default=1, help='Radius scaler to scale the sigma of the blob and use the result as a radius of the blob. Default: 1')

args = parser.parse_args()

min_sigma = args.min_sigma
max_sigma = args.max_sigma
threshold = args.threshold
radius_scaler = args.radius_scaler

# %%
def detect_file_path():
    print("Detecting file path...")
    file_path = Path.cwd() / 'Red Sea depth.asc'
    return file_path

# %%
def read_data(data_path):
    print("Reading data...")
    data = np.loadtxt(data_path, skiprows=6)
    data_16bit = data.astype(np.int16)
    return data_16bit

# %%
def filter_islands(data):
    data = np.where(data > 0, np.nan,  data )
    return data

def filter_edges(data):
    data = np.where(abs(data) < 32767, data, np.nan)
    return data

# %%
def filter_data(data_16bit):
    print("Filtering data...")
    data_16bit = filter_islands(data_16bit)
    data_16bit = filter_edges(data_16bit)
    return data_16bit
# %%
def extract_blobs(data_16bit):
    print("Extracting blobs...")
    blobs = feature.blob_log(data_16bit, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    return blobs


def plot_blobs(data_16bit, blobs, radius_scaler):
    print("Plotting blobs...")
    fig, ax = plt.subplots()
    ax.imshow(data_16bit, cmap='gray')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), radius_scaler*r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
    plt.show()


# main function
if __name__ == "__main__":
    print("Running dome detection with the following parameters: ")
    print("min_sigma: ", min_sigma)
    print("max_sigma: ", max_sigma)
    print("threshold: ", threshold)
    print("radius_scaler: ", radius_scaler)
    data_path = detect_file_path()
    print("Detected data path: ", data_path)
    data = read_data(data_path)
    print("Data read successfully")
    filtered_data = filter_data(data)
    print("Data filtered successfully")
    blobs = extract_blobs(filtered_data)
    print("Blobs extracted successfully, count of blobs: ", len(blobs))
    plot_blobs(filtered_data, blobs, radius_scaler)