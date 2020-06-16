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
from keras.callbacks import EarlyStopping

#1. data
np.random.seed()

train_path = './data/train'
test_path = './data/test'

# image_generator
datagen = ImageDataGenerator(zoom_range = [0.5, 1.0],          # 랜덤하게 줌, 아웃
                             rescale = 1. / 255,
                             brightness_range = [0.2, 1.0],    # 이미지 발기
                             rotation_range = 90,              # 랜덤한 각도로 돌리기
                             horizontal_flip = True,           # 위, 아래, 오른쪽, 왼쪽 뒤집기
                             vertical_flip = True,             
                             height_shift_range = 0.3,         # 위, 아래 움직임 
                             width_shift_range = 0.3,          # 오른쪽, 왼쪽 움직임
                             validation_split= 0.2
                            )



x_train = datagen.flow_from_directory( train_path,
                                target_size = (256, 256),
                                batch_size = 10,                # batch_size만큼 랜덤하게 변형된 이미지 획득
                                subset="training",
                                class_mode = 'categorical')

x_val = datagen.flow_from_directory( train_path,
                                target_size = (256, 256),
                                batch_size = 10,                
                                subset="validation",
                                class_mode = 'categorical')

x_test = datagen.flow_from_directory( test_path,
                                target_size = (256, 256),
                                batch_size = 10,                
                                class_mode = 'categorical')

#2. model
model = Sequential()
model.add(Conv2D(50, (3, 3), input_shape= ( 256, 256, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Conv2D(100, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Conv2D(100, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Conv2D(100, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
# model.add(Conv2D(100, (3, 3), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = 2))
# model.add(Dropout(0.3))
# model.add(Conv2D(100, (3, 3), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = 2))
# model.add(Dropout(0.3))
# model.add(Conv2D(100, (3, 3), padding = 'same', activation = 'relu'))
# model.add(Dropout(0.3))
model.add(Conv2D(50, (3, 3), padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(9, activation = 'softmax'))

es = EarlyStopping(monitor = 'val_loss', patience = 100)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit_generator(x_train, epochs = 500, 
                           steps_per_epoch= 2,                       # 한 세대하다 몇번 생성기로부터 데이터를 얻을 것이가                 
                           validation_data= x_val,                   # validation_data 설정
                           validation_steps= 100,
                           callbacks= [es])

# model.save
model.save('./model_train1.h5')

loss, acc =model.evaluate_generator(x_test, steps = 2)
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

# loss_graph
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.',c= 'blue', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()

plt.show()

# predict
y_pred = model.predict_generator(x_test[0])
print(x_test[0])
print(y_pred)





    