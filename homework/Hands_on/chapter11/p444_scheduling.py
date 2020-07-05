'''
# Learning Schedule(학습 스케줄)
: 큰 학습률로 시작하고 학습 속도가 느려질 때 학습률을 낮추면 최적의 고정 학습률보다 좋은 솔루션을 더 빨리 발견할 수 있음
: 훈련하는 동안 학습률을 감소시키는 전략 = 학습 스케줄
'''
from tensorflow import keras

#1. 거듭제곱 기반 스케줄링(power scheduling)
# : 학습률을 반복 횟수 t에 대한 함수 n(t) = n0 / (1+ t/s)^c 로 지정 
# : n0 = 초기 학습률 / c = 거듭 제곱 수(일반적으로 1) / s = 스텝 횟수
optimizer = keras.optimizers.SGD(lr = 0.01, decay = 1e-4)


#--------------------------------------------------------------------------------------------
#2. 지수 기반 스케줄링(exponential scheduling)
# : 학습률이 s 스텝마다 10배씩 점차 줄어듬
def exponential_decay_fn(epoch):
    return 0.01 * 0.1 **(epoch / 20)

def exponential_decay(lr0, s):         # 초기 학습률, 스텝 성정 가능
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0 = 0.01, s = 20)

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn) 
                                # 스케줄링 함수를 전달하여 LearningRateScheduler콜백 만듦
                                # LearningRateScheduler는 epoch를 시작할 때 마다 optimiszer의 lr 속성을 업뎃함

history = model.fit(x_train_scaled, y_train, [...], callbacks = [lr_scheduler])

def exponential_decay_fn(epoch, lr):    # optimizer의 초기 학습률에만 의존함
    return lr * 0.1 ** (1 / 20)             


#---------------------------------------------------------------------------------------------
#3. 구간별 고정 스케줄링(piecewise constant scheduling)
# : 일정 횟수의 epoch동안 일정한 학습률을 사용, 그다음 또 다른 횟수의 epoch동안 작음 학습률 사용
# ex) 5 epochs 동안 lr = 0.1 / 50 epochs 동안 lr = 0.001   
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001

lr_scheduler = keras.callbacks.ReduceLROnRlateau(factor = 0.5, patience = 5)
                                                 # 최상의 검증손실이 5번 연속적인 epoch동안 향상되지 않을 때마다 lr에 0.5곱함

# 위와 동일한 지수 기반 스케일링 구현 
s = 20 * len(x_train) // 32                      # 20번 epoch에 담긴 전체 스텝 수 (batch_size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate) 


#------------------------------------------------------------------------------------------------
#4. 성능 기반 스케줄링(performance scheduling)
# : 매 N 스텝마다 (조기 종료처럼) 검증 오차를 측정하고 오차가 줄어들지 않으면 lamda배 만큼 학습률 감소시킴


#------------------------------------------------------------------------------------------------
#5. 1사이클 스케줄링(1cycle scheduling)
# : 훈련 절반 동안 초기 학습률 n0을 선형적으로 n1까지 증가시킴