import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

groups_floder_path = 'D:/data/train/'
categories = categories = ["carrot","chicken","egg","fish",'flour',"mashroom","meat", "onion", "paprika","potato"]
num_classes = len(categories)

image_w = 100
image_h = 50


X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_floder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir+filename)
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
            X.append(img/256)     # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

print(X.shape)   # (163, 50, 100, 3)
print(Y.shape)   # (163, 9)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 33, train_size = 0.8)

xy = (x_train, x_test, y_train, y_test)
np.save("./mini_preject/graph/img_data.npy", xy)

plt.imshow(X[0], 'gray')
plt.show()