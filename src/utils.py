import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from keras.utils.np_utils import to_categorical

def get_data(folder):
	"""
	Load the data and labels from the given folder.
	"""
	X = []
	y = []
	for wbc_type in os.listdir(folder):
		if not wbc_type.startswith('.'):
			if wbc_type in ['NEUTROPHIL']:
				label = 0
			elif wbc_type in ['EOSINOPHIL']:
				label = 1
			elif wbc_type in ['MONOCYTE']:
				label = 2
			elif wbc_type in ['LYMPHOCYTE']:
				label = 3
			else:
				label=4
			for image_filename in tqdm(os.listdir(folder + wbc_type)):
				img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
				if img_file is not None:
					img_file = np.array(Image.fromarray(img_file).resize((80, 60)))
					img_arr = np.asarray(img_file)
					X.append(img_arr)
					y.append(label)
	X = np.asarray(X)
	y = np.asarray(y)
	y_Hot = to_categorical(y , num_classes = 5)
	return X,y_Hot