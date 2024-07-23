import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xml
import xml.etree.ElementTree as ET
import cv2
import math
import warnings
from skimage.transform import resize
import os
from scipy import ndimage
import shutil

def create_subwindows_and_counts(image, numpy_folder, stride = 8, kernel_size = 32):
    im_name = image.split(".")[0]
    im_file = im_name + '.npy'
    count_file = im_name + '_density_map.npy'
    # load the image and the count numpy files
    loaded_im_file = np.load(numpy_folder + '/' + im_file)
    loaded_count_file = np.load(numpy_folder + '/' + count_file)
        
    # create the subwindows and counts as follows
    img_height = loaded_im_file.shape[0]
    img_width = loaded_im_file.shape[1]
    
    density_sums = []
    catch_image = []
    for i in  range(0, img_height, stride):
        for j in range(0, img_width, stride):
            sub_window = loaded_im_file[i: i + kernel_size, j : j + kernel_size,:]
            density = loaded_count_file[i: i + kernel_size, j : j + kernel_size]
            dense_sum = np.sum(density)
            density_sums.append(dense_sum)
            sub_window = resize(sub_window, (32, 32,3))
            catch_image.append(sub_window)

    # save the combined subwindows and counts
    return(catch_image,density_sums, im_file)

# let's do this for a sample and then in the loop
train_files_path = "Preprocessed_train_data/all_img_density_files/"

train_im_and_map_contents = os.listdir(train_files_path)

# sort these - ALWAYS sort these as the order is always messed up on HCC
train_im_and_map_contents.sort()
print(len(train_im_and_map_contents))
# get only the names of the image (npy files)
train_im_names = [item for item in train_im_and_map_contents if item.split(".")[0][-3:] != 'map']
print(len(train_im_names))

# create the subwindows for all train data
catch_all_image_subwindows_train = []
catch_all_dense_subwindows_train = []
catch_train_names = []
for image in train_im_names:
    train_ims, train_maps, train_names = create_subwindows_and_counts(image, train_files_path, stride = 8, kernel_size = 32)
    catch_all_image_subwindows_train.append(train_ims)
    catch_all_dense_subwindows_train.append(train_maps)
    catch_train_names.append(train_names)

print(np.mean(train_im_names == catch_train_names))

# stack the images
train_im_stack = np.vstack(catch_all_image_subwindows_train)
print(train_im_stack.shape)

# stack the subcounts
train_count_stack = np.hstack(catch_all_dense_subwindows_train)
print(train_count_stack.shape)

# do a little more sanity checks to make sure the stacking is correctly done
# for images
index = 127
for i in range(index):
    print(np.mean(train_im_stack[6144*index:6144+6144*index,:,:,:] == catch_all_image_subwindows_train[index]), np.mean(train_count_stack[6144*index:6144+6144*index,] == catch_all_dense_subwindows_train[index]))

# need to save these files
train_save_path = 'final_train_sub_windows_and_counts'

# save the sub images
np.save(train_save_path + "/" + "train_sub_windows.npy", train_im_stack)
# save the sub counts
np.save(train_save_path + "/" + "train_sub_counts.npy", train_count_stack)