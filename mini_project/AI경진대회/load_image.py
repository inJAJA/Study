import numpy as np
import cv2
import os
import glob
from fourier_trans import fourier_trans
import re

# os로 불러오기 : 파일 형식 상관 X
def load_image_os(image_dir):
    X = []
    # resize 크기
    image_w = 100
    image_h = 50

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            # print(image_dir+filename)
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
            X.append(img/256)     # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
    
    return np.array(X)



# glob으로 불러오기 : 지정한 파일 형식만 불러오기
def load_image(image_dir):
    X = []
    # resize 크기
    image_w = 150
    image_h = 150

    image_dir = glob.glob(image_dir+'*.jpg')

    for filename in image_dir:
        print(filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
        X.append(img/255)     # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
        print(img.shape)
    
    return np.array(X)

# label 텍스트 순서로 불러옴
def load_image_fourier(file, path):  
    X = []
    # resize 크기
    image_w = 56
    image_h = 56

    f = open(file, 'r')
    txt = f.read().splitlines()
    # print(txt)
    fold = '/tf/notebooks/Ja/data/'+path+'/'
    for filename in txt:
        title = filename.replace(' 0', '').replace(' 1', '')
        img = cv2.imread(fold+str(title), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)

        # print(img)
        img_back = fourier_trans(img)
        X.append(img_back/255)     # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
    
    return np.array(X)


train_path = '/tf/notebooks/Ja/data/train/train_label.txt'
test_path = '/tf/notebooks/Ja/data/test/test_label.txt'
val_path = '/tf/notebooks/Ja/data/validate/validate_label.txt'


# x_train = load_image_fourier(train_path, 'train')
# x_test = load_image_fourier(test_path, 'test')
x_val = load_image_fourier(val_path, 'validate')
# print(x_val.shape)



# np.save('/tf/notebooks/Ja/data/x_train.npy', arr = x_train) 
# np.save('/tf/notebooks/Ja/data/x_test.npy', arr = x_test) 
np.save('/tf/notebooks/Ja/data/x_val.npy', arr = x_val) 

print('save complete')