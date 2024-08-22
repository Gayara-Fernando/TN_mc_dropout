# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import shutil
from scipy import ndimage
# from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pickle

# load trained model here
TN_model = tf.keras.models.load_model('models/TN_model_with_new_data.h5')
# look at the input shape
TN_model.input

# Let's redefine the model
new_input = tf.keras.layers.Input(shape = (32,32,3))

# add a conv2d layer
new_conv2d_1 = tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same')
new_conv2d_1_out = new_conv2d_1(new_input)
# add activation
new_act_1 = tf.keras.layers.Activation("relu")
new_act_1_out = new_act_1(new_conv2d_1_out)

# add a conv2d layer
new_conv2d_2 = tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'valid')
new_conv2d_2_out = new_conv2d_2(new_act_1_out)
# add activation
new_act_2 = tf.keras.layers.Activation("relu")
new_act_2_out = new_act_2(new_conv2d_2_out)

# add a maxpooling layer
max_pooling_1 = tf.keras.layers.MaxPool2D()
max_pooling_1_out = max_pooling_1(new_act_2_out)

# add a dropout layer
dropout_1 = tf.keras.layers.Dropout(0.05)
dropout_1_out = dropout_1(max_pooling_1_out, training = True)

# add a conv2d layer
new_conv2d_3 = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')
new_conv2d_3_out = new_conv2d_3(dropout_1_out)
# add activation
new_act_3 = tf.keras.layers.Activation("relu")
new_act_3_out = new_act_3(new_conv2d_3_out)

# add a conv2d layer
new_conv2d_4 = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'valid')
new_conv2d_4_out = new_conv2d_4(new_act_3_out)
# add activation
new_act_4 = tf.keras.layers.Activation("relu")
new_act_4_out = new_act_4(new_conv2d_4_out)

# add a maxpooling layer
max_pooling_2 = tf.keras.layers.MaxPool2D()
max_pooling_2_out = max_pooling_2(new_act_4_out)

# add a dropout layer
dropout_2 = tf.keras.layers.Dropout(0.05)
dropout_2_out = dropout_2(max_pooling_2_out, training = True)

# add a flatten layer
flat_layer = tf.keras.layers.Flatten()
flat_layer_out = flat_layer(dropout_2_out)

# add a dense layer
dense_1 = tf.keras.layers.Dense(512)
dense_1_out = dense_1(flat_layer_out)
# add activation
new_act_5 = tf.keras.layers.Activation("relu")
new_act_5_out = new_act_5(dense_1_out)
# add a dropout layer
dropout_3 = tf.keras.layers.Dropout(0.05)
dropout_3_out = dropout_3(new_act_5_out, training = True)

# add a dense layer - prediction head
dense_final = tf.keras.layers.Dense(1)
dense_final_out = dense_final(dropout_3_out)
# add activation
new_act_6 = tf.keras.layers.Activation("relu")
new_act_6_out = new_act_6(dense_final_out)

new_model = tf.keras.models.Model(inputs = new_input, outputs = new_act_6_out)

weights = TN_model.get_weights()
new_model.set_weights(weights)
weights_new = new_model.get_weights()
print(len(weights), len(weights_new))

for i in range(len(weights)):
    print(np.mean(weights[i] == weights_new[i]))

block_number = 'Block_13'
file_path  = 'Preprocessed_test_data/all_img_density_files/'
split_test_file_path = os.path.join(file_path, block_number)

split_test_images = [file for file in os.listdir(split_test_file_path) if file.split(".")[0][-3:] != 'map']
split_test_images.sort()

# check the shapes of the image files
counter = 0
for file in split_test_images:
    load_file = np.load(split_test_file_path + '/' + file)
    counter = counter + 1
    print(counter, file, load_file.shape)

# get the stacked test sub windows path
block_number = 'Block_13'
stacked_test_file_path = os.path.join('final_test_sub_windows_and_counts/', block_number)
stacked_test_file_path

# get the contents
contents_stacked_test_path = os.listdir(stacked_test_file_path)
contents_stacked_test_path.sort()

# get only the image files
images_stacked_test = [file for file in contents_stacked_test_path if file[:8] == 'test_ims']
images_stacked_test.sort()

# define a function to get the post-hoc prediction
def prediction_on_test_data(model, numpy_folder, v_stack_folder, selected_file, stride = 8, kernel_size = 32):
#     load the cnn model
    
# load the image data file
    load_image = np.load(numpy_folder + "/"+ selected_file)
    
    # get the image height
    img_height = load_image.shape[0]
    # get the image weight
    img_width = load_image.shape[1]

    selected_stacked_file = 'test_ims_' + selected_file
    all_test_sub_windows = np.load(v_stack_folder + "/"+ selected_stacked_file)

    # now, to get the predictions, pass the sub windows
    test_image_prediction = model.predict(all_test_sub_windows)
    
    # density map
    Density_map = np.zeros((img_height, img_width))

    # counts map
    counts_map = np.zeros((img_height, img_width))
    
    # now, for every window, we will keep adding the values together and also add the counts
    counter = 0
#     need a counter to move into each predicted value in the pred values list
    for ii in range(0, img_height, stride):
        for jj in range(0, img_width, stride):
#         operations for density map
#             get the window of interest
            new_window = Density_map[ii:ii + kernel_size,jj:jj+kernel_size]
#     fill each with the value c_k
            counts_window = np.full((new_window.shape[0], new_window.shape[1]), test_image_prediction[counter])
#     get the shapes of this new window
            cw_height = counts_window.shape[0]
            cw_width = counts_window.shape[1]
#         Do c_k/r_2
            counts_window_new = counts_window/(cw_height*cw_width)
#     This is the value in the window now
            value_window = counts_window_new
#     place the values in the corrsponding area of the density map
            Density_map[ii:ii + kernel_size,jj:jj+kernel_size] = new_window + value_window

#         Let's now focus on capturing the counts of the windows
            new_window_c = counts_map[ii:ii + kernel_size,jj:jj+kernel_size]
#     get the counts area
            count = np.ones((new_window_c.shape[0], new_window_c.shape[1]))
#     keep adding the counts to reflect the addition of densities
            counts_map[ii:ii + kernel_size,jj:jj+kernel_size] = new_window_c + count
#     increase the counter
            counter = counter + 1
            
#         get the normalized count
    normalized_counts = np.divide(Density_map, counts_map)
    
#     entire count on the test set
    pred_on_test = np.sum(normalized_counts)
    
#     return the predicted value
    return(pred_on_test, normalized_counts, selected_file)

with open('Calibration_model/NN_model.pkl', 'rb') as f:
    NN_model = pickle.load(f)

# get the predictions for test data

# save density map path
dense_path = "predicted_count_maps_for_test_files/Block_13"

final_values_preds_names = []
final_values_calibrated_preds_names = []
for file in split_test_images:
    name = file
    all_preicted_values = []
    all_calibrated_predicted_values = []
    # normalized_pred_maps = []
    for i in range(25):
        preds_value, norm_counts, _ = prediction_on_test_data(new_model, split_test_file_path, stacked_test_file_path, file, stride = 8, kernel_size = 32)
        # np.array(6.5).reshape(1,-1)
        calibrated_preds = NN_model.predict(np.array(preds_value).reshape(1,-1))
        # save the normalized density maps
        np.save(dense_path + '/' + file.split('.')[0] + '_' + str(i) + '_norm_map_TN.npy', norm_counts)
        all_preicted_values.append(preds_value)
        all_calibrated_predicted_values.append(np.float32(calibrated_preds))
        # normalized_pred_maps.append(norm_counts)
    final_values_preds_names.append((name, all_preicted_values))
    final_values_calibrated_preds_names.append((name, all_calibrated_predicted_values))

preds_df = pd.DataFrame(final_values_preds_names, columns = ['image_name', 'predictions'])
preds_df = preds_df.join(preds_df['predictions'].apply(pd.Series))
preds_df = preds_df.drop(['predictions'], axis = 1)
preds_df.columns = ['name'] + ['predicted_count_' + str(i) for i in range(25)]
preds_df['name'] = [i.split(".")[0] for i in preds_df['name']]

calibrated_preds_df = pd.DataFrame(final_values_calibrated_preds_names, columns = ['image_name', 'predictions'])
calibrated_preds_df = calibrated_preds_df.join(calibrated_preds_df['predictions'].apply(pd.Series))
calibrated_preds_df = calibrated_preds_df.drop(['predictions'], axis = 1)
calibrated_preds_df.columns = ['name'] + ['calib_predicted_count_' + str(i) for i in range(25)]
calibrated_preds_df['name'] = [i.split(".")[0] for i in calibrated_preds_df['name']]

# import the true test counts
true_counts_df = pd.read_csv("True_tassel_counts/test_data/true_test_counts_blk_13.csv")

# merge the two dataframes
joined_df = pd.merge(true_counts_df, preds_df, on = 'name')
joined_df_calibrated = pd.merge(true_counts_df, calibrated_preds_df, on = 'name')

only_preds = preds_df.iloc[:,1:]

print("Uncalibrated results")
li_test = np.percentile(only_preds, axis = 1, q = (2.5, 97.5))[0,:].reshape(-1,1)     
ui_test = np.percentile(only_preds, axis = 1, q = (2.5, 97.5))[1,:].reshape(-1,1)   
width_test = ui_test - li_test
avg_width_test = width_test.mean(0)[0]
print(avg_width_test)
li_and_ui = np.hstack((li_test, ui_test))
# save the li and ui
li_and_ui_df = pd.DataFrame(li_and_ui, columns = ['lower_limit', 'upper_limit'])
# add the names of the images
li_and_ui_df_final = pd.concat((preds_df["name"], li_and_ui_df), axis = 1)
# save this dataframe for future use
li_and_ui_df_final.to_csv("True_tassel_counts/test_data/test_predicted_intervals_TN_model_blk_13.csv", index = False)
y_true = true_counts_df['true_count'].values.reshape(-1,1)
ind_test = (y_true >= li_test) & (y_true <= ui_test)
coverage_test= ind_test.mean(0)[0]
print(coverage_test)
averaged_preds = only_preds.mean(axis = 1)
mae = mean_absolute_error(true_counts_df['true_count'], averaged_preds)
print("mae uncalibrated: ", mae)
rmse = np.sqrt(mean_squared_error(true_counts_df['true_count'], averaged_preds))
print("rmse uncalibrated: ", rmse)
pearson_corr = pearsonr(true_counts_df['true_count'], averaged_preds)
print("pearson uncalibrated: ", pearson_corr.statistic)
r2_score_val = r2_score(true_counts_df['true_count'], averaged_preds)
print("R2 uncalibrated: ", r2_score_val)

print("Calibrated results")

only_preds_calibrated = calibrated_preds_df.iloc[:,1:]
li_test_calibrated = np.percentile(only_preds_calibrated, axis = 1, q = (2.5, 97.5))[0,:].reshape(-1,1)     
ui_test_calibrated = np.percentile(only_preds_calibrated, axis = 1, q = (2.5, 97.5))[1,:].reshape(-1,1)   
width_test_calibrated = ui_test_calibrated - li_test_calibrated
avg_width_test_calibrated = width_test_calibrated.mean(0)[0]
print(avg_width_test_calibrated)
li_and_ui_calibrated = np.hstack((li_test_calibrated, ui_test_calibrated))
# save the li and ui
li_and_ui_df_calibrated = pd.DataFrame(li_and_ui_calibrated, columns = ['lower_limit', 'upper_limit'])
# add the names of the images
li_and_ui_df_calibrated = pd.concat((calibrated_preds_df["name"], li_and_ui_df_calibrated), axis = 1)
y_true = true_counts_df['true_count'].values.reshape(-1,1)
ind_test_calibrated = (y_true >= li_test_calibrated) & (y_true <= ui_test_calibrated)
coverage_test_calibrated= ind_test_calibrated.mean(0)[0]
print(coverage_test_calibrated)
averaged_preds_calibrated = only_preds_calibrated.mean(axis = 1)
mae = mean_absolute_error(true_counts_df['true_count'], averaged_preds_calibrated)
print("mae calibrated: ",mae)
rmse = np.sqrt(mean_squared_error(true_counts_df['true_count'], averaged_preds_calibrated))
print("rmse calibrated: ", rmse)
pearson_corr = pearsonr(true_counts_df['true_count'], averaged_preds_calibrated)
print("corr calibrated: ", pearson_corr.statistic)
r2_score_val = r2_score(true_counts_df['true_count'], averaged_preds_calibrated)
print("r2 calibrated: ", r2_score_val)
plt.figure(figsize = (10,10))
plt.scatter( averaged_preds_calibrated, true_counts_df['true_count'])
plt.title("Scatter plot for true vs predicted values")
plt.xlabel("Averaged predicted counts")
plt.ylabel("True counts")
ind = str(13_0.05)
name = 'block_' + ind + '.pdf' 
location = os.path.join('Calibration_model/plots/', name)
plt.savefig(location, format="pdf", bbox_inches="tight")