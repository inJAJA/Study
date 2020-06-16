from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt 
import glob
import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

np.random.seed()

path = './data/train'

# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range = [0.5, 1.0],
                             rescale = 1. / 255,
                            #  brightness_range = [0.2, 1.0],
                             rotation_range = 90, 
                             horizontal_flip = True,
                             vertical_flip = True, 
                             height_shift_range = 0.3,
                             width_shift_range = 0.3,
                            )
i = 0

for train_gen in datagen.flow_from_directory( path,
                                         target_size = (256, 256),
                                         batch_size = 10,
                                         class_mode = 'categorical',
                                         save_to_dir= './data/image_gen',     # 저장 경로
                                         save_prefix= 'image',                        # 파일명
                                         save_format = 'jpg'
                                         ):                        # 파일형식
    i += 1
    if i > 10:
        break

