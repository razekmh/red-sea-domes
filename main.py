# %%
import numpy as np
from skimage import feature, measure
import matplotlib.pyplot as plt
from pathlib import Path
import numpy.ma as ma


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
    # check the count of negative value vs positive values
    # # if negative values are more then remove the positive values
    # # if positive values are more then remove the negative values

    # # count the number of negative values
    # neg_count = np.count_nonzero(data < 0)
    # pos_count = np.count_nonzero(data > 0)

    # if neg_count > pos_count:
        # remove positive values
    #     data = np.where(data < 0, data, np.nan)
    # else:
    #     # remove negative values
    #     data = np.where(data > 0, data, np.nan)
    data = np.where(data > 0, np.nan,  data )
    return data
# %%
def filter_edges(data):
    data = np.where(abs(data) < 32767, data, np.nan)
    return data

# %%
def filter_data(data_16bit):
    print("Filtering data...")
    # data_16bit = np.multiply(data_16bit, -1)
    data_16bit = filter_islands(data_16bit)
    data_16bit = filter_edges(data_16bit)
    return data_16bit
# %%
def extract_blobs(data_16bit):
    print("Extracting blobs...")
    blobs = feature.blob_log(data_16bit, min_sigma=20, max_sigma=50, threshold=0.01)
    #print(len(blobs))
    return blobs

# %%
def perimeter_calc(contour):
    assert contour.ndim == 2 and contour.shape[1] == 2, contour.shape
 
    shift_contour = np.roll(contour, 1, axis=0)
    dists = np.sqrt(np.power((contour - shift_contour), 2).sum(axis=1))
    return dists.sum()
# %%
def get_first_countours(blobs, data_16bit, blobs_count_limit=None):
    print("Extracting first countours...")
    if blobs_count_limit == None:
        blobs_count_limit = len(blobs)
    
    print("Extracting "+str(blobs_count_limit)+" blobs...")

    first_countour_list = []
    blob_counter = 0
    
    for blob in blobs:
        if len(first_countour_list) >= blobs_count_limit:
            return first_countour_list
        y, x, r = blob
        # Find the contour of the blob
        mask = np.zeros_like(data_16bit, dtype=np.uint16)
        expand_size = 5
        start_y, end_y = int(y - r - expand_size), int(y + r + 1 + expand_size)
        start_x, end_x = int(x - r - expand_size), int(x + r + 1 + expand_size)
        mask[start_y:end_y, start_x:end_x] = 1
        filled_data = np.nan_to_num(data_16bit)
        masked_data = filled_data * mask
        masked_data[masked_data == 0] = np.nan
        contour = measure.find_contours(masked_data)[0]
        first_countour_list.append(contour)
        if blob_counter % 100 == 0:
            print("Count of blobs checked: ",blob_counter)

        blob_counter += 1
    return first_countour_list
   

# %%
def classify_blobs(blobs, data_16bit, first_countour_list):
    print("Classifying blobs...")
    # Initialize lists to store circularities and non-circular blobs
    circularities = []
    non_circular_blobs = []
    circular_blobs = []
    blob_counter = 0
    for blob_index in range(len(blobs)):

        # Calculate circularity
        # print(measure.moments(contour))
        area = measure.moments(first_countour_list[blob_index])[0,0]  # Corrected line
        perimeter = perimeter_calc(first_countour_list[blob_index])
        # perimeter = measure.perimeter(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularities.append(circularity)

        # Set a threshold for circularity to classify non-circular blobs
        circularity_threshold = 700
        if circularity < circularity_threshold:
            non_circular_blobs.append(blobs[blob_index])
        else:
            circular_blobs.append(blobs[blob_index])
        blob_counter += 1
        if blob_counter % 100 == 0:
            print("count of blobs processed: ",blob_counter)

    print("count of circular_blobs: ", len(circular_blobs))
    print("count of non_circular_blobs: ", len(non_circular_blobs))
    return circular_blobs, non_circular_blobs

# %%
def plot_blobs(data_16bit,non_circular_blobs,circular_blobs, first_countour_list):
    print("Plotting blobs...")
    # Display the detected and non-circular blobs (for visualization)
    fig, ax = plt.subplots()
    ax.imshow(data_16bit, cmap='gray')

    for blob in non_circular_blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    for blob in circular_blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
        ax.add_patch(c)

    # plot first contour
    # for one_contour in first_countour_list:
    #     for contour in one_contour:
    #         ax.plot(contour[:, 1], contour[:, 0], color='green', linewidth=2)


    plt.show()

# %%
# main function
# if __name__ == "__main__":
#     data_path = detect_file_path()
#     print("Detected data path: ", data_path)
#     data = read_data(data_path)
#     print("Data read successfully")
#     filtered_data = filter_data(data)
#     print("Data filtered successfully")
#     blobs = extract_blobs(filtered_data)
#     print("Blobs extracted successfully, count of blobs: ", len(blobs))
#     circular_blobs, non_circular_blob, first_countour_list = classify_blobs(blobs, filtered_data)
#     print("Blobs classified successfully")
#     plot_blobs(filtered_data,non_circular_blobs,circular_blobs)
# %%
data_path = detect_file_path()
print("Detected data path: ", data_path)
# %%
data = read_data(data_path)
print("Data read successfully")
# %%
filtered_data = filter_data(data)
print("Data filtered successfully")
# %%
blobs = extract_blobs(filtered_data)
print("Blobs extracted successfully, count of blobs: ", len(blobs))
# %%
# first_countour_list = get_first_countours(blobs, filtered_data)#, blobs_count_limit=1)
# # %%
# circular_blobs, non_circular_blobs = classify_blobs(blobs, filtered_data, first_countour_list)
# print("Blobs classified successfully")
# %%
# plot_blobs(filtered_data,non_circular_blobs,circular_blobs,first_countour_list)
# # %%


# fig, ax = plt.subplots()
# ax.imshow(filtered_data, cmap='gray')
# for contour in first_countour_list:
#         contour = contour[0]
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
# plt.show()


# count_of_blobs = 0
# mask = np.zeros_like(filtered_data, dtype=np.uint16)
# for blob in blobs:
#     y, x, r = blob
#     mask[int(y - r):int(y + r + 1), int(x - r):int(x + r + 1)] = 1
#     count_of_blobs += 1
#     if count_of_blobs > 50:
#         break 

# fig, ax = plt.subplots()
# ax.imshow(mask, cmap='gray')
# plt.show()
# %%
# fig, ax = plt.subplots()
# ax.imshow(filtered_data, cmap='gray')
# for one_contour in first_countour_list:
#     for contour in one_contour:
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
# plt.show()
# %%
fig, ax = plt.subplots()
ax.imshow(filtered_data, cmap='gray')
for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
plt.show()
