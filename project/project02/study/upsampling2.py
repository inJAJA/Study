from keras.models import Model
from keras.layers import Input, UpSampling2D
import numpy as np

x = np.array([[10, 20],[30, 40]])
print(x)                            
# [[10 20]
#  [30 40]]
x = x[None, ..., None]                                    
print(x.shape)                      # (1, 2, 2, 1)


# Upsampling
def upsample(interpolation = None):
    inp = Input(shape = (2, 2, 1))
    up = UpSampling2D(interpolation = interpolation)(inp)

    model = Model(inp, up)
    return model


# result : 1 
result = upsample('nearest').predict(x)
print(result.shape)                         # (1, 4, 4, 1)

result = result.reshape(4, 4)
print(result)
                                            # [[10. 10. 20. 20.]
                                            #  [10. 10. 20. 20.]
                                            #  [30. 30. 40. 40.]
                                            #  [30. 30. 40. 40.]]


# result : 2
result = upsample('bilinear').predict(x)
print(result.shape)                         # (1, 4, 4, 1) 

result = result.reshape(4, 4)
print(result)
                                            # [[10. 15. 20. 20.]
                                            #  [20. 25. 30. 30.]
                                            #  [30. 35. 40. 40.]
                                            #  [30. 35. 40. 40.]]

