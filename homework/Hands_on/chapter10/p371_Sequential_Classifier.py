import tensorflow as tf 
from tensorflow import keras 
import numpy as np 

#1. data
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full),(x_test, y_test) = fashion_mnist.load_data()

print(x_train_full.shape)   # (60000, 28, 28)
print(x_train_full.dtype)   # uint8

x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test / 255.0

class_names = ["T-shirt/top",'Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

print(class_names[y_train[0]])  # Coat

#2. model01
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

#2. model02
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dense(300, activation = 'relu'),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.summary()

# layers
print(model.layers)
# [<tensorflow.python.keras.layers.core.Flatten object at 0x000001DFB79A1308>, 
# <tensorflow.python.keras.layers.core.Dense object at 0x000001DFB79D1188>, 
# <tensorflow.python.keras.layers.core.Dense object at 0x000001DFB79D1488>, 
# <tensorflow.python.keras.layers.core.Dense object at 0x000001DFB79D1808>]

hidden1 = model.layers[1]                     # model에 있는 layer를 인덱스로 쉽게 가져 올 수 있다.
print(hidden1.name)                           # dense_3

print(model.get_layer('dense_3') is hidden1)  # True


# weight : 층의 모든 파라미터는 get_weights()메서드와 set_weights()메서를 이용하여 접근 할 수 있다.
weights, biases = hidden1.get_weights() 
print(weights)          # [[ 0.03161622 -0.03172589 -0.02751372 ... -0.0657685  -0.00146628   
                        #   -0.02791729]
                        #  ...
                        #  [-0.04660155  0.03536018 -0.06368248 ...  0.03875828  0.04986876   
                        #    0.03702278]]

print(weights.shape)    # (784, 300)

print(biases)           # [0. 0. 0. 0. 0. 0. 0. 0. .... 0. 0. 0. 0. 0. 0. 0. 0. 0.]

print(biases.shape)     # (300,)

'''
# Dense층은 연결 가중치를 무작위로 초기화, 편향은 0으로 초기화
# 다른 초기화 방법을 사용하고 싶으면 kernel_initializer, bias_initializer 매개변수 설정 가능
'''

#3. compile
model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = 'sgd',
              metrics = ['accuracy']) 


#3. fit
history = model.fit(x_train, y_train, epochs = 30,
                    validation_data = (x_valid, y_valid))

'''
# 어떤 클래스는 많이 등장되고 다른 클래스는 조금 등장하여 train_set가 편중되어 있을 때
# class_weight  : 적게 등장하는 클래스에 높은 가중치 부여, 많이 등장하는 클래스에 낮은 가중치
# sample_weight : 샘플별로 가중치 부여
# (두 매겨변수가 모두 지정되면 keras는 두 값을 곱하여 사용) 
'''

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # 수직축의 범위는 [0-1] 사이로 설정
plt.show()
                    

#4. evaluate
print(model.evaluate(x_test, y_test))   # [0.32290190387964246, 0.8855]

#4. predict
x_new = x_test[:3]
y_proba = model.predict(x_new)
print(y_proba.round(2))                 # [[0.   0.   0.   0.   0.   0.   0.   0.01 0.   0.99]
                                        #  [0.   0.   1.   0.   0.   0.   0.   0.   0.   0.  ]
                                        #  [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]]

y_pred = model.predict_classes(x_new)   # .predict_classes : 가장 높은 확률을 가진 클래스에만 관심이 있다면
print(y_pred)                           # [9 2 1]

print(np.array(class_names)[y_pred])    # ['Ankle boot' 'Pullover' 'Trouser']