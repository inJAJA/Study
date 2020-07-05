'''
# BatchNormalization (BN)
: over fitting 방지법
: 그라디언트 소실과 폭주 문제를 해결하기 위해 사용 
: 각 층에서 활성화 함수를 통과하기 전이나 후에 모델에 연산을 하나 추가

: 입력을 원적에 맞추고 '정규화'한 다음, 각 층에서 두개의 새로운 파라미터로 결과값의 스케일을 조정, 이동함
: 신경망의 첫 번째 층으로 배치 정규화를 추가하면 train_set를 (예를 들어 StandardScaler를 사용하여) 표준화할 필요 X
'''
from tensorflow import keras

# model01
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation = 'elu', kernel_initializer = 'he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation = 'elu', kernel_initializer = 'he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation = 'softmax')
])

model.summary()
# BN 층은 입력마다 4개의 파라미터를 추가함

# 첫번째 BN 층의 파라미터 살펴보기
print([(var.name, var.trainable) for var in model.layers[1].variables])
# [('batch_normalization/gamma:0', True), 
# ('batch_normalization/beta:0', True),                       # 두 개는 (역전파로) 훈련되고 
# ('batch_normalization/moving_mean:0', False), 
# ('batch_normalization/moving_variance:0', False)]           # 두 개는 훈련되지 않음

print(model.layers[1].updates)                                 
# [<tf.Operation 'cond/Identity' type=Identity>, 
# <tf.Operation 'cond_1/Identity' type=Identity>]
# keras에서 BN 층 만들 때, 훈련하는 동안 매 반복마다 keras에서 호출된 두 개의 연산이 함께 생선됨 
# : 이 연산이 평균을 업데이트 (tensorflow backend연산)


# model02
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer = 'he_normal', use_bias = False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu'),                            
    keras.layers.Dense(100, kernel_initializer = 'he_normal', use_bias = False),
    keras.layers.BatchNormalization(),                      
    keras.layers.Activation('elu'),
    keras.layers.Dense(10, activation = 'softmax')
])
# BN 논문의 저자들은 활성화 함수 이전에 BN 층을 추가하는 것이 좋다고 조언 
# 작업에 때라 선호되는 방식이 다름 - 주어진 데이터셋에 무엇이 더 좋은지 알아서 판단

# BN층은 입력마다 이동 param을 포함하기 때문에 이전 층에서 bias을 뺄 수 있음(use_bias = False 사용)