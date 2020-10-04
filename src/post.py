import os
import random
import shutil
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import myplotstyle
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator,\
        array_to_img, img_to_array, load_img


def plot_sample_distribution():
    n_lymph, n_nonlymph = 34, 321
    x1 = 1
    plt.figure()
    plt.bar(np.arange(x1, 2*x1 + 0.1, x1), [n_lymph, n_nonlymph])
    plt.text(x1, n_lymph + 1.0, str(n_lymph), ha='center')
    plt.text(2*x1, n_nonlymph + 1.0, str(n_nonlymph), ha='center')
    plt.ylabel('Image Sample Count')
    plt.ylim([0, n_nonlymph + 20])
    plt.xticks(np.arange(x1, 2*x1 + 0.1, x1), ['LYMPHCYTE', 'NON LYMPHCYTE'])
    plt.savefig('../figs/sample_distribution.png')
    plt.show()

def main():
    plot_sample_distribution()

if __name__ == "__main__":
    main()
[1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1]
