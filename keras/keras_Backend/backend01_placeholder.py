from keras import backend as K

'''
# placeholder
: 입력 placeholder를 인스턴스화하는 코드
 = tf.placeholder()
 = th.tensor.matrix()    # th = Theano
 = th.tensor.tensor()
'''
inputs = K.placeholder(shape = (2, 4, 5))
# also works:
inputs = K.placeholder(shape = (None, 4, 5))
# also works:
inputs = K.placeholder(ndim = 3)