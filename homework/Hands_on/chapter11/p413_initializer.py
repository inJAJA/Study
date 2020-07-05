'''
1. 글로럿 : 활성화 함수 없음, 하이퍼볼릭 탄젠트, 소프트맥스
2. He    : ReLU 함수와 그 변종들
3. 르쿤   : SELU

: kreas는 기본적으로 균등분포의 글로럿 초기화를 사용
=> kernel_initializer = 'he_uniform' 이나 'he_normal' 로 바꾸어 He초기화 사용 가능
'''
from tensorflow import keras
keras.layers.Dense(10, activation ='relu', kernel_initializer = 'he_normal')


# fan_in대신에 fan_out기반의 균등분포 He초기화를 사용하고 싶다면 다음과 같이 Variance Scaling사용 가능
he_avg_init = keras.initializers.VarianceScaling(scale = 2., mode = 'fan_avg',
                                                 distribution = 'uniform')
keras.layers.Dense(10, activation = 'relu', kernel_initializer = he_avg_init)