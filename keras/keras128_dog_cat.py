from keras.applications import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

img_dog = load_img('./data/dog_cat/dog.jpg', target_size =(224, 224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size =(224, 224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size =(224, 224))
img_yang = load_img('./data/dog_cat/yang.jpg', target_size =(224, 224))

plt.imshow(img_cat)
# plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

print(arr_dog)
# print(type(arr_dog))
# print(arr_dog.shape)


# RGB -> BGR : standardscaler형식
from keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)

print(arr_dog)                          # (224, 224, 3)
print(arr_dog.shape)

# 이미지 데이터를 하나로 합친다.
import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])

print(arr_input.shape)                  # (4, 224, 224, 3)


#2. model
model = VGG16()
probs = model.predict(arr_input)

print(probs)
print('probs.shape :', probs.shape)     # probs.shape : (4, 1000) 예측 가능 이미지 천개

# 이미지 결과
from keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)

print('----------------------------')
print(results[0])
print('----------------------------')
print(results[1])
print('----------------------------')
print(results[2])
print('----------------------------')
print(results[3])
