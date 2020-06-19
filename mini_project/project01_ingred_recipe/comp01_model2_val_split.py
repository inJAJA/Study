from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 
import numpy as np
import glob
import cv2
import os 

#1. data
np.random.seed()

train_path = 'D:/data/train'
test_path = 'D:/data/test'

# image_generator : validation_split사용
datagen = ImageDataGenerator(zoom_range = [0.5, 1.0],          # 랜덤하게 줌, 아웃
                             rescale = 1. / 255,
                             brightness_range = [0.2, 1.0],    # 이미지 발기
                             rotation_range = 90,              # 랜덤한 각도로 돌리기
                             horizontal_flip = True,           # 위, 아래 뒤집기
                             vertical_flip = True,             # 오른쪽, 왼쪽 뒤집기
                             height_shift_range = 0.3,         # 위, 아래 움직임 
                             width_shift_range = 0.3,          # 오른쪽, 왼쪽 움직임
                             
                             validation_split= 0.2
                            )

x_train = datagen.flow_from_directory( train_path,              
                                target_size = (256, 256),
                                batch_size = 40,               # batch_size만큼 랜덤하게 변형된 이미지 획득
                                subset="training",
                                class_mode = 'categorical')

x_val = datagen.flow_from_directory( train_path,              # train data 이미지 변형하여 사용
                                target_size = (256, 256),
                                batch_size = 25,                
                                subset="validation", 
                                class_mode = 'categorical')

x_test = datagen.flow_from_directory( test_path,
                                target_size = (256, 256),
                                batch_size = 25,                
                                class_mode = 'categorical')

#2. model
model = Sequential()
model.add(Conv2D(50, (5, 5), input_shape= ( 256, 256, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 5))
model.add(Dropout(0.3))
model.add(Conv2D(100, (5, 5), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 5))
model.add(Dropout(0.3))
model.add(Conv2D(100, (5, 5), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 5))
model.add(Dropout(0.3))
model.add(Conv2D(50, (5, 5), padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()

es = EarlyStopping(monitor = 'loss', patience = 50)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit_generator(x_train, epochs = 500, 
                           steps_per_epoch= 50,                       # 한 세대하다 몇번 생성기로부터 데이터를 얻을 것이가                 
                           validation_data= x_val,                   # validation_data 설정
                           validation_steps= 10,                    # [validation data수/배치사이즈]
                           callbacks= [es],
                           verbose = 2)

# model.save
model.save('./model_train2.h5')

loss, acc =model.evaluate_generator(x_test, steps = 10, verbose = 2)
print('loss, acc: ', loss, acc)

# loss_graph
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.',c= 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.legend()

plt.show()







    