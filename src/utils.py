import os
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

def get_annotation(image, tree):
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                    if name[0] == "R":
                        cv2.rectangle(image, (xmin, ymin),
                                (xmax, ymax), (0, 255, 0), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 1)
                    if name[0] == "W":
                        cv2.rectangle(image, (xmin, ymin),
                                (xmax, ymax), (0, 0, 255), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
                    if name[0] == "P":
                        cv2.rectangle(image, (xmin, ymin),
                                (xmax, ymax), (255, 0, 0), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 0), 1)
    plt.figure(figsize=(16,16))
    plt.imshow(image)
    plt.show()

def crop_wbc(image, tree):
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                    if name[0] == "W":
                        return xmin, ymin, xmax, ymax

def augment_image(path, img_size):
    data_gen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    return data_gen.flow_from_directory(path, target_size=(img_size, img_size),
            class_mode="categorical", batch_size=64, shuffle=True)

def gen_masked_img():
    fname = "../dataset-master/labels.csv"
    df = pd.read_csv(fname)
    df = df[["Image", "Category"]]
    df.dropna(inplace=True)
    labels = pd.Series(df.Category.values, index=df.Image).to_dict()

    # Note that the function below is adapted from https://github.com/Shenggan/BCCD_Dataset
    for i in range(0, 500):
        image_index = "BloodImage_00{:03d}".format(i)
        try :
            lab = labels[i]
            if "," in lab:
                print(f"Image {i}, multi-labels, skipped.")
                continue
            image = cv2.imread(f"../BCCD_Dataset/BCCD/JPEGImages/{image_index}.jpg")
            tree = ET.parse(f"../BCCD_Dataset/BCCD/Annotations/{image_index}.xml")
            new_image = np.zeros_like(image)
            wbc = crop_wbc(image, tree)
            new_image[wbc[1]:wbc[3], wbc[0]:wbc[2], :] = 1
            new_image = np.multiply(new_image, image)
            cv2.imwrite(f"../dataset-master/masked/{lab}/BloodImage_{i}.jpg", new_image)
        except :
            continue

def main():
    print(augment_image("../dataset-master/masked/", 100))

if __name__ == "__main__":
    main()
