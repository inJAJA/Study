from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam
 
                  # 출력 레이어를 포함시킬 건가 말건가
takemodel = VGG16(include_top=False)   # input = (None, 224, 224, 3)
# takemodel = VGG19()
# takemodel = ResNet50()
# takemodel = InceptionV3()
# takemodel = MobileNetV2()


takemodel.summary()

model = Sequential()
model.add(takemodel)      
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation= 'softmax'))

model.summary()
'''
input_tensor = Input(shape=( 32, 32, 3))
vgg16 = VGG16(include_top = False, weights = 'imagenet', input_tensor = input_tensor)

top_model = vgg16.output
top_model = Faltten(input_shape = vgg16, output_shape[1:])(top_model)
top_model = Dense(26, activation = 'sigmoid')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(10, activation = 'softmax')(top_model)

model = Model(inputs = vgg16.input, outputs = top_model)
'''

# [참고] https://keras.io/api/applications/

