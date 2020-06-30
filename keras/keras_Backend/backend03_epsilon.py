import keras.backend as K

# epsilon
e = K.epsilon()
print(e)                  # 1e-07


# set_epsilon
set_e = K.set_epsilon(e)  # 수치 식에 사용되는 fuzz factor의 값을 설정합니다.


# example
e0 = K.epsilon()
K.set_epsilon(1e-05)       # epsilon값 설정
e1 = K.epsilon()

print(e0)                  # 1e-07
print(e1)                  # 1e-05
