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
import matplotlib.pyplot as plt
import random
import cv2
from imgaug import augmenters as iaa


# 
# Setting all the global parameters
# 

im_height = 1000
im_width = 1000
channels = 4
patch_size_image = 125
patch_size_model = 128
height = patch_size_image
width = patch_size_image
stride = 40


# 
# Defining paths to different image folders
# 

# Base path:
base_path = '/////'

# Pre-processed Images directory path:
im_path = base_path + 'tissue_images/'
# Pre-processed Target directory path:
target_path = base_path + 'target/'
# Pre-processed Target where 50% border is removed, directory path:
target_border_rem_50_path = base_path + 'target_border_remove_50/'
# Pre-processed Target where 70% border is removed, directory path:
target_border_rem_70_path = base_path + 'target_border_remove_70/'
# Processed original size images path:
im_processed_path = base_path + 'tissue_images_macenko_only_on_18-5592/'
# H Stain images path (sklearn rgb2hed)
h_stain_path = base_path + 'tissue_images_h_extracted_method_1/'
# hs(V) value of macenko normalized
mecenko_hsv_gray_path = base_path + 'tissue_images_macenko_only_on_18-5592_gray/'
# hs(V) value of original images
im_hsv_gray_path = base_path + 'tissue_images_original_gray/'


# 
# Reading the normalized tissue images stored on disk and storing them as array
# 

all_files_temp = os.listdir(target_path)
all_files = []
for file in tqdm(all_files_temp):
	if file.strip().split('.')[-1] in ['png']:
		all_files.append(''.join(file.strip().split('.')[0:-1]))

original_images = np.zeros((len(all_files), im_height, im_width, channels), dtype=np.uint8)

for im_i in tqdm(range(len(all_files))):
	image_file = im_processed_path + all_files[im_i] + '.png'
	original_images[im_i, :, :, :3] = skimageio.imread(image_file)
	
for im_i in tqdm(range(len(all_files))):
	image_file = h_stain_path + all_files[im_i] + '.png'
	original_images[im_i, :, :, 3] = (skimageio.imread(image_file)/256).astype(np.uint8)


# 
# Reading the original masks stored on disk and storing them as array
# 

original_masks = np.zeros((len(all_files), im_height, im_width), dtype=np.bool)
for im_i in tqdm(range(len(all_files))):
	mask_file = target_border_rem_50_path + all_files[im_i] + '.png'
	original_masks[im_i] = skimageio.imread(mask_file)


# 
# Creating patches of defined size from the full size images/masks and storing them as array (with fixed strides and no augmentation)
# 

num_files = len(all_files)
num_images = int(num_files * (np.floor((im_height - height) / stride) + 2) ** 2)

print('Number of files: %d' % num_files)
print('Number of images total: %d' % num_images)

images = np.zeros((num_images, patch_size_image, patch_size_image, channels), dtype=np.uint8)
masks = np.zeros((num_images, patch_size_image, patch_size_image, 1), dtype=np.bool)

print('Reading original images and masks and preparing the input data')
counter = 0
for im_i in tqdm(range(len(all_files))):
	image = original_images[im_i]
	mask = original_masks[im_i]
	for i in list(range(0, im_height-height+1, stride))+[im_height-height]:
		for j in list(range(0, im_width-height+1, stride))+[im_height-height]:
			images[counter,:,:,:] = image[i:i+height, j:j+width, :]
			# images[counter,:,:,:] = skimageresize(image[i:i+height, j:j+width, :], (rs_height, rs_width), mode='constant', cval=0, preserve_range=True)
			masks[counter,:,:,0] = mask[i:i+height, j:j+width]
			counter += 1


# 
# Creating patches of defined size from the full size images/masks and storing them as array (by randomly picking a patch from the full image and with augmentation)
# 

num_files = len(all_files)
# original_per_image = 500
# augmented_1_per_image = 300
augmented_2_per_image = 300 # CLAHE
# augmented_3_per_image = 300
augmented_4_per_image = 300
augmented_5_per_image = 100
augmented_6_per_image = 200
# augmented_7_per_image = 150
num_images_original = int(num_files * (np.floor((im_height - height) / stride) + 2) ** 2)
num_images_augmented = (augmented_2_per_image +
					   augmented_6_per_image + 
					   augmented_4_per_image + 
					   augmented_5_per_image) * num_files
num_images = num_images_original + num_images_augmented

print('Number of files: %d' % num_files)
print('Number of images original: %d' % num_images_original)
print('Number of images augmented: %d' % num_images_augmented)
print('Number of images total: %d' % (num_images_original + num_images_augmented))

images = np.zeros((num_images, patch_size_image, patch_size_image, channels), dtype=np.uint8)
masks = np.zeros((num_images, patch_size_image, patch_size_image, 1), dtype=np.bool)

counter = 0

print('Reading original images and masks and preparing the input data')
counter = 0
for im_i in tqdm(range(len(all_files))):
	image = original_images[im_i]
	mask = original_masks[im_i]
	for i in list(range(0, im_height-height+1, stride))+[im_height-height]:
		for j in list(range(0, im_width-height+1, stride))+[im_height-height]:
			images[counter,:,:,:] = image[i:i+height, j:j+width, :]
			# images[counter,:,:,:] = skimageresize(image[i:i+height, j:j+width, :], (rs_height, rs_width), mode='constant', cval=0, preserve_range=True)
			masks[counter,:,:,0] = mask[i:i+height, j:j+width]
			counter += 1

seq = iaa.Sequential([iaa.Fliplr(0.15), iaa.Flipud(0.15)])
seq_det = seq.to_deterministic()
images[:counter, :, :, :] = seq_det.augment_images(images[:counter, :, :, :])
masks[:counter, :, :, :] = seq_det.augment_images(masks[:counter, :, :, :])

# print(counter)

# print('Augmentation 1 : Channel random shuffle')
# for im_i in tqdm(range(len(all_files))):
#	 image = original_images[im_i]
#	 mask = original_masks[im_i]
#	 y = random.sample(range(0, im_height-patch_size_image), augmented_per_image)
#	 x = random.sample(range(0, im_width-patch_size_image), augmented_per_image)
#	 for i in range(augmented_per_image):
#		 image_patch = image[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image, :]
#		 mask_patch = mask[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image]
#		 image_patch2 = np.take(image_patch[:,:,:3], np.random.permutation(image_patch[:,:,:3].shape[2]), axis=2) # Channel shuffle
#		 images[counter, :, :, :3] = image_patch2
#		 images[counter, :, :, 3] = image_patch[:, :, 3]
#		 masks[counter, :, :, 0] = mask_patch
#		 counter += 1
		
# print(counter)

print('Augmentation 2 : CLAHE')
def augment_clahe(img):
	img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)
	return img_adapteq

for im_i in tqdm(range(len(all_files))):
	image = original_images[im_i]
	mask = original_masks[im_i]
	y = random.sample(range(0, im_height-patch_size_image), augmented_2_per_image)
	x = random.sample(range(0, im_width-patch_size_image), augmented_2_per_image)
	for i in range(augmented_2_per_image):
		image_patch = image[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image, :]
		mask_patch = mask[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image]
		image_patch2 = augment_clahe(image_patch[:, :, :3])
		images[counter, :, :, :3] = image_patch2
		images[counter, :, :, 3] = image_patch[:, :, 3]
		masks[counter, :, :, 0] = mask_patch
		counter += 1
		
# print(counter)

# print('Augmentation 3 : AdditiveGaussianNoise')
# augmenter3 = iaa.AdditiveGaussianNoise(loc=0.0, scale=(0, 0.25*255), per_channel=False)

# for im_i in tqdm(range(len(all_files))):
#	 image = original_images[im_i]
#	 mask = original_masks[im_i]
#	 y = random.sample(range(0, im_height-patch_size_image), augmented_per_image)
#	 x = random.sample(range(0, im_width-patch_size_image), augmented_per_image)
#	 for i in range(augmented_per_image):
#		 image_patch = image[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image, :]
#		 mask_patch = mask[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image]
#		 images[counter, :, :, :] = augmenter3.augment_image(image_patch)
#		 masks[counter, :, :, 0] = mask_patch
#		 counter += 1
		
# print(counter)

print('Augmentation 4 : GaussianBlur')
augmenter4 = iaa.GaussianBlur(1.0)

for im_i in tqdm(range(len(all_files))):
	image = original_images[im_i]
	mask = original_masks[im_i]
	y = random.sample(range(0, im_height-patch_size_image), augmented_4_per_image)
	x = random.sample(range(0, im_width-patch_size_image), augmented_4_per_image)
	for i in range(augmented_4_per_image):
		image_patch = image[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image, :]
		mask_patch = mask[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image]
		images[counter, :, :, :] = augmenter4.augment_image(image_patch)
		masks[counter, :, :, 0] = mask_patch
		counter += 1
		
# print(counter)

print('Augmentation 5 : Affine Rotate')
for im_i in tqdm(range(len(all_files))):
	image = original_images[im_i]
	mask = original_masks[im_i]
	y = random.sample(range(0, im_height-patch_size_image), augmented_5_per_image)
	x = random.sample(range(0, im_width-patch_size_image), augmented_5_per_image)
	for i in range(augmented_5_per_image):
		angle = random.choice([90,180,270])
		augmenter5 = iaa.Affine(rotate=angle, mode='reflect')
		image_patch = image[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image, :]
		mask_patch = mask[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image]
		images[counter, :, :, :] = augmenter5.augment_image(image_patch)
		masks[counter, :, :, 0] = augmenter5.augment_image(mask_patch)
		counter += 1
		
# print(counter)

# print('Augmentation 6 : Affine Scale')
# for im_i in tqdm(range(len(all_files))):
#	 image = original_images[im_i]
#	 mask = original_masks[im_i]
#	 y = random.sample(range(0, im_height-patch_size_image), augmented_6_per_image)
#	 x = random.sample(range(0, im_width-patch_size_image), augmented_6_per_image)
#	 for i in range(augmented_6_per_image):
#		 scale = random.choice(list(range(70,90)) + list(range(110,130)))*1.0/100
#		 augmenter6 = iaa.Affine(scale=scale, mode='reflect')
#		 image_patch = image[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image, :]
#		 mask_patch = mask[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image]
#		 images[counter, :, :, :] = augmenter6.augment_image(image_patch)
#		 masks[counter, :, :, 0] = augmenter6.augment_image(mask_patch)
#		 counter += 1
		
# print(counter)

print('Augmentation 6 : Affine Scale')
for im_i in tqdm(range(len(all_files))):
	image = original_images[im_i]
	mask = original_masks[im_i]
	for i in range(augmented_6_per_image):
		scale = random.choice(list(range(60,90)) + list(range(110,140)))*1.0/100
		new_patch_size = int(patch_size_image / scale)
		y = random.choice(range(0, im_height-new_patch_size))
		x = random.choice(range(0, im_width-new_patch_size))
		image_patch = image[y:y+new_patch_size, x:x+new_patch_size, :]
		mask_patch = mask[y:y+new_patch_size, x:x+new_patch_size]
		images[counter, :, :, :] = skimageresize(image_patch, (patch_size_image, patch_size_image), mode='constant', cval=0, preserve_range=True)
		masks[counter, :, :, 0] = skimageresize(mask_patch, (patch_size_image, patch_size_image), mode='constant', cval=0, preserve_range=True)
		counter += 1
		
# print(counter)

# print('Augmentation 7 : Affine Shear')
# for im_i in tqdm(range(len(all_files))):
#	 image = original_images[im_i]
#	 mask = original_masks[im_i]
#	 y = random.sample(range(0, im_height-patch_size_image), augmented_per_image)
#	 x = random.sample(range(0, im_width-patch_size_image), augmented_per_image)
#	 for i in range(augmented_7_per_image):
#		 angle = random.choice(range(-10,10))
#		 augmenter7 = iaa.Affine(shear=angle, mode='reflect')
#		 image_patch = image[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image, :]
#		 mask_patch = mask[y[i]:y[i]+patch_size_image, x[i]:x[i]+patch_size_image]
#		 images[counter, :, :, :] = augmenter7.augment_image(image_patch)
#		 masks[counter, :, :, 0] = augmenter7.augment_image(mask_patch)
#		 counter += 1


# 
# Changing the size of the image patches to match the input size to the model
# 

images2 = np.zeros((num_images, patch_size_model, patch_size_model, channels), dtype=np.uint8)
for i in tqdm(range(num_images)):
	images2[i, :, :, :] = skimageresize(images[i,:,:,:], (patch_size_model, patch_size_model), mode='constant', cval=0, preserve_range=True)
	
images = images2
del images2


# 
# Changing the size of the mask patches to match the input size to the model
# 

masks2 = np.zeros((num_images, patch_size_model, patch_size_model, 1), dtype=np.uint8)
for i in tqdm(range(num_images)):
	masks2[i, :, :, 0] = skimageresize(masks[i,:,:,0], (patch_size_model, patch_size_model), mode='constant', cval=0, preserve_range=True)
	
masks = masks2
del masks2


# 
# Convert the image/mask patches from uint8/bool to float32
# 

masks = masks.astype(np.float32)
images = images.astype(np.float32)


# 
# Rescaling the image patches to (0-1) by dividing by 255
# 

images /= 255


# 
# Some checks
# 

# Print the shape of the input image patches
print(images.shape)

# Print the shape of the target mask patches
print(masks.shape)

# Print the max and min of the input image patches array
print(np.max(images))
print(np.min(images))

# Print the unique values of the output mask patches array
print(np.unique(masks[0]))


# 
# Defining and training a U-net model
# 

IMG_WIDTH = IMG_HEIGHT = 128
IMG_CHANNELS = 4
NUM_EPOCHS = 30
BATCH_SIZE = 8
STEPS_PER_EPOCHS = 1800

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

model = keras.models.Model(inputs=[inputs], outputs=[conv_10])

optimizer = keras.optimizers.Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early = keras.callbacks.EarlyStopping(patience=3, verbose=1)
checkpoint = keras.callbacks.ModelCheckpoint(base_path + '/models/keras_unet.h5', save_best_only=True)

# Print model summary
model.summary()

# Fit the model
result = model.fit(images, 
				   masks,
				   verbose=1,
				   validation_split=0.1,
				   batch_size=BATCH_SIZE, 
				   epochs=NUM_EPOCHS, 
				   callbacks=[early, checkpoint],
				  shuffle=True)


# 
# Saving the model after training is finished
# 

model.save(base_path + '/models/keras_unet.h5')