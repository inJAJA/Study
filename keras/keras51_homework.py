# keras49_classify2.py
# ome_hot_encoding
# 무조건 인덱스의 시작이 0부터 시작된다.
# 1.앞에 0을 자르는 방법
# - 슬라이싱사용 
# - sk.learn의 one_hot_encoding사용
# - 판다스 사용



# 선생님의 답 1 : keras의 to_categorical 사용

# x = [1, 2, 3]
# x = x - 1
# print(x)                                    # TypeError : list + int

import numpy as np
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) 
y = y - 1                                     # numpy에서만 가능
                                              # 단점 : 한가지 자료형만 써야 한다.
print(y)                                      # y = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

from keras.utils import np_utils              # shape가 0부터 시작이 된다.
y = np_utils.to_categorical(y)
print(y)
print(y.shape) 
"""
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
 (10, 5)
"""                              



# 선생님의 답 2
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) 
print(y.shape)                                 # (10, )
# y = y.reshape(-1, 1)                         # sklearn의 OneHotEncoder을 사용하기 위해서는 2차원이어야 한다.
y = y.reshape(10, 1)                           # 위듸 코드와 동일


from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()                 # sklearn은 그 숫자만큼 생성


print(y)
print(y.shape)
"""
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
 (10, 5)
"""
# np.argmax() : 제일 큰 값의 자리의 수를 가져온다.
y_pred = np.argmax(y, axis =1)                 # axis =1 , 행별로 최댓값을 뺀다.
print(y_pred)                                  # y = [0 1 2 3 4 0 1 2 3 4]
                     
y1_pred = np.argmax(y, axis=1 )+1           
print(y1_pred)                                 # y1 =[1 2 3 4 5 1 2 3 4 5]


