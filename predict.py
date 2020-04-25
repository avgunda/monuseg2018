# 
# Importing all the necessary libraries
# 

import os
import numpy as np
import skimage
from skimage import io as skimageio
from skimage.transform import resize as skimageresize
import keras
import random
from keras.preprocessing import image as image_utils
from tqdm import tqdm
from keras.models import load_model
from skimage import color
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa
import scipy
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage import feature


# 
# Reading the full size validation images and masks
# 

base_path = '/////'

im_path = base_path + '/final_data/tissue_images_macenko_only_on_18-5592/test/'
im_path2 = base_path + '/final_data/tissue_images_h_extracted_method_1/test/'
im_path3 = base_path + '/final_data/tissue_images_macenko_only_on_18-5592_gray/validate/'
target_path = base_path + '/final_data/target/validate/'

all_files_temp = os.listdir(im_path)
all_files = []
for file in tqdm(all_files_temp):
	if file.strip().split('.')[-1] in ['png']:
		all_files.append(''.join(file.strip().split('.')[0:-1])) 

original_images = np.zeros((len(all_files), 1000, 1000, 4), dtype=np.uint8)
	
for im_i in tqdm(range(len(all_files))):
	original_images[im_i, :, :, :3] = skimageio.imread(im_path+all_files[im_i]+'.png')
	original_images[im_i, :, :, 3] = (skimageio.imread(im_path2+all_files[im_i]+'.png')/256).astype(np.uint8)


# 
# Loading the saved model from disk (which was trained on aws)
# 

model = load_model(base_path + '/models/final/unet1.h5')


# 
# Resizing the valition images to 1024 size and storing as array
# 

original_images_2 = np.zeros((len(original_images), 1024, 1024, 4), dtype=np.uint8)

for i in tqdm(range(len(original_images))):
	original_images_2[i] = skimageresize(original_images[i], (1024, 1024), mode='constant', cval=0, preserve_range=True)


# 
# Defining a full-size U-net model
# 

IMG_WIDTH = IMG_HEIGHT = 1024
IMG_CHANNELS = 4

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
# s = Lambda(lambda x: x / 255) (inputs)

conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_1)
conv_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling2D((2, 2))(conv_1)

conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_2)
conv_2 = BatchNormalization()(conv_2)
pool_2 = MaxPooling2D((2, 2))(conv_2)

conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3)
conv_3 = BatchNormalization()(conv_3)
pool_3 = MaxPooling2D((2, 2))(conv_3)

conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_3)
conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_4)
conv_4 = BatchNormalization()(conv_4)
pool_4 = MaxPooling2D((2, 2))(conv_4)

conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_4)
dropout = Dropout(0.2)(conv_5)
conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(dropout)
conv_5 = BatchNormalization()(conv_5)

up_6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_5)
up_6 = concatenate([up_6, conv_4], axis=3)
conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up_6)
conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_6)
conv_6 = BatchNormalization()(conv_6)

up_7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_6)
up_7 = concatenate([up_7, conv_3], axis=3)
conv_7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up_7)
conv_7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_7)
conv_7 = BatchNormalization()(conv_7)

up_8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_7)
up_8 = concatenate([up_8, conv_2], axis=3)
conv_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up_8)
conv_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_8)
conv_8 = BatchNormalization()(conv_8)

up_9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_8)
up_9 = concatenate([up_9, conv_1], axis=3)
conv_9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up_9)
conv_9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_9)
conv_9 = BatchNormalization()(conv_9)

# TODO: Drop-out layers at the end of the contracting path perform further implicit data augmentation.

conv_10 = Conv2D(1, (1, 1), activation='sigmoid')(conv_9)

model_new = keras.models.Model(inputs=[inputs], outputs=[conv_10])


# 
# Transfer the weights of the small input size model to full size model
# 

model_new.set_weights(model.get_weights())


# 
# Rescaling the validation full size normalized tissue images to (0-1) for input into model
# 

original_images_2 = original_images_2.astype(np.float32)
original_images_2 /= 255


# 
# Initializing array for the full size prediction masks
# 

predicted_masks = np.zeros((len(all_files), 1000, 1000), dtype=np.float32)


# 
# Creating the path where the prediction masks are to be stored
# 

prediction_path = base_path + '/predictions/final/unet1_tta_045/'
if not os.path.exists(prediction_path):
	os.makedirs(prediction_path)


# 
# (TTA - Test time augmentation)
# Predicting patch mask using model and resizing and creating the full size prediction mask
# 

for im_i in tqdm(range(len(all_files))):
	# im_i=0
	input_image = original_images_2[im_i]
	output = model_new.predict(np.expand_dims(input_image, axis=0))[0, :, :, 0]
	#
	aug1 = iaa.Fliplr(1.0)
	input2 = aug1.augment_image(input_image)
	output2 = model_new.predict(np.expand_dims(input2, axis=0))[0, :, :, 0]
	output2 = aug1.augment_image(output2)
	#
	aug2 = iaa.Flipud(1.0)
	input3 = aug2.augment_image(input_image)
	output3 = model_new.predict(np.expand_dims(input3, axis=0))[0, :, :, 0]
	output3 = aug2.augment_image(output3)
	#
	input4 = aug1.augment_image(aug2.augment_image(input_image))
	output4 = model_new.predict(np.expand_dims(input4, axis=0))[0, :, :, 0]
	output4 = aug1.augment_image(aug2.augment_image(output4))
	#
	aug4 = iaa.Affine(rotate=90)
	aug4_rev = iaa.Affine(rotate=-90)
	input5 = aug4.augment_image(input_image)
	output5 = model_new.predict(np.expand_dims(input5, axis=0))[0, :, :, 0]
	output5 = aug4_rev.augment_image(output5)
	#
	aug5 = iaa.Affine(rotate=270)
	aug5_rev = iaa.Affine(rotate=-270)
	input6 = aug5.augment_image(input_image)
	output6 = model_new.predict(np.expand_dims(input6, axis=0))[0, :, :, 0]
	output6 = aug5_rev.augment_image(output6)
	#
	aug6 = iaa.Sequential([iaa.Affine(rotate=270), iaa.Fliplr(1.0)])
	aug6_rev = iaa.Sequential([iaa.Fliplr(1.0), iaa.Affine(rotate=-270)])
	input7 = aug6.augment_image(input_image)
	output7 = model_new.predict(np.expand_dims(input7, axis=0))[0, :, :, 0]
	output7 = aug6_rev.augment_image(output7)
	#
	aug7 = iaa.Sequential([iaa.Affine(rotate=90), iaa.Fliplr(1.0)])
	aug7_rev = iaa.Sequential([iaa.Fliplr(1.0), iaa.Affine(rotate=-90)])
	input8 = aug7.augment_image(input_image)
	output8 = model_new.predict(np.expand_dims(input8, axis=0))[0, :, :, 0]
	output8 = aug7_rev.augment_image(output8) 
	#
	output_mask = (output + output2 + output3 + output4 + output5 + output6 + output7 + output8) / 8
	output_mask = skimageresize(output_mask, (1000, 1000), mode='constant', cval=0, preserve_range=True)
	temp1 = output_mask > 0.45
	# 
	predicted_masks[im_i] = temp1.astype(np.float32)


# 
# Saving the final prediction images to disk
# 

for im_i in tqdm(range(len(all_files))):
	skimage.io.imsave(prediction_path+all_files[im_i]+'.png', predicted_masks[im_i])


