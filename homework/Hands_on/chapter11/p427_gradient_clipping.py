'''
# Gradient clipping (그레이디언트 클리핑)
: 그레이디언트 폭주 문제를 완화하는 인기있는 방법
: 역전파 될 때 일정 임곗값을 넘어서지 못하게 그레이디언트를 잘라내는 것
# 방법 : optimizer만들 때 clipvalue & clipnorm 매개 변수를 지정
'''
from tensorflow import keras 

optimizer = keras.optimizzers.SGD(clipvalue = 1.0) # gradient 벡터의 모든 원소를 -1.0과 1.0 사이로 clipping
model.compile(loss = 'mse', optimizer = optimizer) # = (훈련되는 각 파라미터에 대한) 손실의 모든 편미분 값을 -1.0에서 1.0로 잘라냄

