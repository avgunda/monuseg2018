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
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import watershed

# 
# Defining the AJI function
# 

def calculate_aji(mask1, mask2):
	# Mask1 >> Prediction mask
	# Mask2 >> Ground truth mask
	if mask1.shape != mask2.shape:
		raise ValueError('The shapes of masks do not match !')
# 	labels1, n1 = ndi.label(mask1) # ndi.label >> 0 is background, anything > 0 is object, no diff betwn 1 & 2
	labels1 = mask1
	n1 = np.max(mask1)	
	labels2, n2 = ndi.label(mask2)
	groups1 = []
	groups2 = []
	for i in range(n1):
		label = i + 1
		coordinates_x = np.where(labels1 == label)[0]
		coordinates_y = np.where(labels1 == label)[1]
		temp = []
		for j in range(len(coordinates_x)):
			temp.append((coordinates_x[j], coordinates_y[j]))
		groups1.append(temp)
	print('Step 1 of 3 completed')
	for j in range(len(groups1)-1, -1, -1):
		if len(groups1[j]) <= 10:
			del groups1[j]
	for i in range(n2):
		label = i + 1
		coordinates_x = np.where(labels2 == label)[0]
		coordinates_y = np.where(labels2 == label)[1]
		temp = []
		for j in range(len(coordinates_x)):
			temp.append((coordinates_x[j], coordinates_y[j]))
		groups2.append(temp)
	print('Step 2 of 3 completed')
	c = 0
	u = 0
	sj = np.zeros(len(groups1), dtype = np.uint8)
	print('Step 3 of 3:')
	for g_nucleus in tqdm(groups2, total=len(groups2)):
		max_intersection = 0
		max_j = -1
		for j in range(len(groups1)):
			if (len(set(g_nucleus) & set(groups1[j])) * 1.0 / len(set(g_nucleus) | set(groups1[j]))) > max_intersection:
				max_j = j
				max_intersection = (len(set(g_nucleus) & set(groups1[j])) * 1.0 / len(set(g_nucleus) | set(groups1[j])))
		if max_j >= 0:
			c += len(set(g_nucleus) & set(groups1[max_j]))
			u += len(set(g_nucleus) | set(groups1[max_j]))
			sj[max_j] = 1
	for j in range(len(sj)):
		if sj[j] == 0:
			u += len(groups1[j])
	return c * 1.0 / u


# 
# Watershed post-processing, Reading the ground truth full size masks and calculate AJI score
# 


aji = 0.0
for im_i in tqdm(range(len(all_files))):
	image = predicted_masks[im_i, :, :]
	distance = ndi.distance_transform_edt(image>0.6)
	local_maxi = peak_local_max(distance, labels=(image > 0.6), min_distance=10,  indices=False) #footprint=np.ones((3, 3)),
	markers = ndi.label(local_maxi)[0]
	image1 = watershed(-distance, markers, mask=image)
	image2 = skimageio.imread(base_path + '/target/validate/'+all_files[im_i]+'.png')
	image2[image2 != 1] = 0 #All the boundaries are set to 0
	aji_score = calculate_aji(image1, image2)
	aji += aji_score
	print(aji_score)

print('The mean AJI score is: %f'%(aji/5))
