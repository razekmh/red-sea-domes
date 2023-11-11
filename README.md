# Red Sea Domes Extraction

## Swtich to this branch and run the script
- Open a new terminal and navigate to the directory of your local repository
- Fetch updates from remote to local using `git fetch`
- Switch to this branch using `git checkout 2-build-utility-to-check-different-sigmas-for-blob-log-detection`
- Run the program using `python main.py` followed by the options you would like to set. The avaiable options are 
```
-ns MIN_SIGMA, --min_sigma MIN_SIGMA
                        Minimum standard deviation for Gaussian kernel. Default: 0
-xs MAX_SIGMA, --max_sigma MAX_SIGMA
                        Maximum standard deviation for Gaussian kernel. Default: 15
-t THRESHOLD, --threshold THRESHOLD
                        Threshold value to detect blobs. Default: 0.01
-rs RADIUS_SCALER, --radius_scaler RADIUS_SCALER
                        Radius scaler to scale the sigma of the blob and use the result as a radius of the
                        blob. Default: 1
```
an example for setting the options would be `python main.py -ns 10 -xs 30 -t 0.1 -rs 2`. You could also choose to set only a subset of the options. For example `python main.py -ns 5 -t 0.1 -rs 2`


## Background
Using the bathymetry data of the read sea floor we aim to extract dome protroding out of the floor.

## Data
Data is provided as a `asc` file. The following are quick stats taken from the header of the asc file.

| Attribute    | Value          |
|--------------|----------------|
| ncols        | 2662           |
| nrows        | 4193           |
| xllcorner    | 32.358333333362|
| yllcorner    | 12.495833333373|
| cellsize     | 0.004166666667 |
| NODATA_value | -32767         |

apart from the header the file can be read as a matrix of digital numbers. Each digital number represent a pixel (cell) of the data and corrosponds to a geographic location. 

Data can be read into a `numpy array` by discarding the header lines.

## Method
The method used to extract the domes is as follows:
1. Read the data into a numpy array
2. Apply a threshold to the data to remove all islands data
3. Replace `NODATA_value` with `numpy.nan`
4. Apply [Laplacian of Gaussian (LoG)](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html#laplacian-of-gaussian-log) to extract blobs (i.e. domes)
5. Apply [find contours](https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html) method to find the outter perimeter of each blob.
6. Calculate the `circularity` of the first contour for each blob and store it in a list. The cirularity is calculated using the following equation `circularity = 4 * np.pi * area / (perimeter ** 2)`
7. Classify the blobs to circular and non-circular based on the `circularity` value. All blobs with `circularity` value greater than the threshold are classified as circular and the rest are classified as non-circular.

## Output
Target output is location of each dome with dimentions and statistics including circularity

Current output is:
- `first_countour_list` which is a list of the first contour for each blob. Each first countor is a list of points in image coordinates. For example the first item is 
```
[1855.5 1332.0,
1855.5 1331.0,
1855.5 1330.0,
1855.5 1329.0,
1855.5 1328.0,
1855.5 1327.0,
1855.5 1326.0,
1855.5 1325.0,
1855.5 1324.0,
1855.5 1323.0,
1855.5 1322.0,
1855.5 1321.0,
1855.5 1320.0,
1855.5 1319.0,
1855.5 1318.0,
1855.5 1317.0,
1855.5 1316.0,
1855.5 1315.0,
1855.5 1314.0,
1855.5 1313.0,
1855.5 1312.0,
1855.0 1311.5,
1854.0 1311.5,
1853.0 1311.5,
1852.0 1311.5,
1851.0 1311.5,
1850.0 1311.5,
1849.0 1311.5,
1848.0 1311.5,
1847.0 1311.5,
1846.0 1311.5,
1845.0 1311.5,
1844.0 1311.5,
1843.0 1311.5,
1842.0 1311.5,
1841.0 1311.5,
1840.0 1311.5,
1839.0 1311.5,
1838.0 1311.5,
1837.0 1311.5,
1836.0 1311.5,
1835.0 1311.5,
1834.5 1312.0,
1834.5 1313.0,
1834.5 1314.0,
1834.5 1315.0,
1834.5 1316.0,
1834.5 1317.0,
1834.5 1318.0,
1834.5 1319.0,
1834.5 1320.0,
1834.5 1321.0,
1834.5 1322.0,
1834.5 1323.0,
1834.5 1324.0,
1834.5 1325.0,
1834.5 1326.0,
1834.5 1327.0,
1834.5 1328.0,
1834.5 1329.0,
1834.5 1330.0,
1834.5 1331.0,
1834.5 1332.0,
1835.0 1332.5,
1836.0 1332.5,
1837.0 1332.5,
1838.0 1332.5,
1839.0 1332.5,
1840.0 1332.5,
1841.0 1332.5,
1842.0 1332.5,
1843.0 1332.5,
1844.0 1332.5,
1845.0 1332.5,
1846.0 1332.5,
1847.0 1332.5,
1848.0 1332.5,
1849.0 1332.5,
1850.0 1332.5,
1851.0 1332.5,
1852.0 1332.5,
1853.0 1332.5,
1854.0 1332.5,
1855.0 1332.5,
1855.5 1332.0]
```

- `non_circular_blobs` and `circular_blobs` both are lists of the blobs. Each blob is presented in the form of a list of center point coordinates in image coordinates and the radius of the blob in image pixels. For example the first non circular blob is presented as `[1845., 1322.,   10.]`

- plot of each blob using center point and radius 

## TODO

- Confirm that the methods of finding `first contour` and `circularity` are correct.
- Convert image coordinates to world coordinates
- Export the results in a format readable by GIS software

## Run locally
- install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- open a new terminal and clone the repo using the following command `git clone https://github.com/razekmh/red-sea-domes.git`
- change directory to the repo `cd red-sea-domes`
- create a new conda environment using the following command `conda env create -f environment.yml`
- activate the new environment `conda activate red-sea-domes`
- run the script `python main.py`
