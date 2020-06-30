from keras import backend as K
import numpy as np

'''
# .variable
 : 변수를 인스턴스화
 = tf.Variable()
 = th.shared()
'''
val = np.random.random((3, 4, 5))  
var = K.variable(value = val)      # 랜덤값 생성
# print(var)


# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# print(var)


# all-ones:
var = K.ones(shape = (3, 4, 5))
# print(var)


# Initializiong Tensors with Random Numbers
b = K.random_uniform_variable(shape = (3, 4), low = 0, high = 1)  # Uniform distribution
c = K.random_normal_variable(shape = (3, 4), mean = 0, scale = 1) # Gaussion distribution
d = K.random_normal_variable(shape = (3, 4), mean = 0, scale = 1)

print(b)
print(c)
print(d)

# Tensor Arithmetic
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis =1)
a = K.softmax(b)
a = K.concatenate([b, c], axis = -1)
