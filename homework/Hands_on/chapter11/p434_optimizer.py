'''
# 훈련속도 높이는 4가지 방법
1. 연결 가중치에 좋은 초기화 전략 사용하기
2. 좋은 활성화 함수 사용하기
3. 배치 정규화 사용
4. 사전 훈련된 네트워크의 일부 재사용 ( 보조 작업 or 비지도 학습을 사용하여 만들 수 있는)
'''
from tensorflow import keras
###고속 옵티마이저

#1. 모멘텀 최적화
# : 경사 하강법 보다 10배 빠르게 진행됨
optimizer = keras.optimizer.SGD(lr = 0.001, momentum = 0.9) # 일반적인 값 0.9


#2. 네스테로프 가속 경사
# : 일반적으로 기본 모멘텀 최적화보다 훈련 속도가 빠름
optimizer = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = True)


#3. AdaGrad
# : 가장 가파른 차원을 따라 gradient 벡터의 스케일을 감소시킴
# : 학습률을 감소시키지만 경사가 완만한 차원보다 가파른 차원에 대해 더 빠르게 감소한다. = 적응적 학습률
# 단점 : 간단한 2차방적식 문제에 대해서는 잘 작동하지만 훈련할 때 너무 일찍 멈추는 경우가 종종 있음
#         => 심층 신경망에는 사용 X


#4. RMSProp
# : (훈련 시작부터의 모든 gradient가 아닌) 가장 최근 반복에서 비롯된 gradient만 누적함
# : Adam이 나오기 전까지 연구자들이 가장 선호하는 최적화 알고리즘이였음
optimizer = keras.optimizers.RMSProp(lr = 0.001, rho = 0.9)


#5. Adam 
# : Adam = adaptive moment estimation(적응적 모멘트 추정)
# : 모멘텀 최적화 + RMSProp
# : 모멘텀 최적화 처럼 gradient의 지수 감소 평균 따르고 / RMSProp처럼 gradient 제곱의 지수 감소된 평균 따름
optimizer = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.9)

#5-1. AdaMax

#5-2. Nadam
# : Adam + 네스테로프 기법
# : 종종 Adam보다 조금 더 빠르게 수렴됨