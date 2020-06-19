from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt 
import glob
import numpy as np
import cv2


np.random.seed()

path = './train_images'

# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range = [0.5, 1.0],
                             rescale = 1. / 255,
                             brightness_range = [0.2, 1.0],
                             rotation_range = 90, 
                             horizontal_flip = True,
                             vertical_flip = True, 
                             height_shift_range = 0.5,
                             width_shift_range = 0.5,
                            )
# 한번에 읽어들이는 장 수
batch_size = 4

# 몇번의 이미지 증식
iterations = 20

# flow_from_directory는 이미지를 불러올 때 폴더명에 맞춰 자동으로 labelling을 해주기 때문입니다.
# flow_from_directory의 next()함수
obj = datagen.flow_from_directory(
     path,
     target_size = (100, 100),
     batch_size = batch_size,
     class_mode = 'categorical')

images = []

for i in enumerate(range(iterations)):
    img, label = obj .next()        # 함수를 한번 호출할 때마다 obj는 설정된 경로에서 batch_size에 맞춰 이미지를 target_size의 크기로 ''형태로 라벨링
    n_img = len(label)

    base = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR) # keras는 RGB, OpenCV는 BGR이라 변경함
    for idx in range(n_img -1):
        img2 = cv2.cvtColor(img[idx + 1], cv2.COLOR_RGB2BGR)
        base = np.hstack((base, img2))
    images.append(base)

img = images[0]
for idx in range(len(images) -1):
    img = np.vstack((img, images[idx + 1]))

cv2.imshow('result',img)