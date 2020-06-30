import keras.backend as K
import numpy as np

# floatx
f = K.floatx()              # default float type을 string으로 반환
print(f)                    # float32


# set_floatx 
K.set_floatx(f)             # default float 타입 설정
                            # floatx : String : 'float16', 'float32', 'float64'


# example
f0 = K.floatx()
K.set_floatx('float16')
f1 = K.floatx()

print(f0)                   # float32
print(f1)                   # float16

print(type(f0))             # <class 'str'>
print(type(f1))             # <class 'str'>


# cast_to_floatx
# K.cast_to_floatx()        # numpy 배열을 keras의 디폴트 float 타입으로 변환
                            # x = numpy 배열


# example
f = K.floatx()
print(f)                    # float16

x = np.array([1.0, 2.0], dtype = 'float64')
print(x.dtype)              # float64

new_x = K.cast_to_floatx(x)
print(new_x)                # [1. 2.]
print(new_x.dtype)          # float16