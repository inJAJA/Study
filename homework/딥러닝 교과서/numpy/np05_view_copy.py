'''
# ndarray의 슬라이스는 배열의 복사본이 아닌 view이다.(list와 다름)
                                         -> 원래의 배열에 영향을 미친다.
'''
import numpy as np

# list의 슬라이스를 이용한 경우
arr_list = [x for x in range(10)]
print('list형 데이터')                           # list형 데이터
print('arr_list: ', arr_list)                   # arr_list:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

arr_list_copy = arr_list[:]
arr_list_copy[0] = 100

print('arr_list: ', arr_list)                   # arr_list:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                                                # 영향 X


# numpy의 ndarray에 슬라이스를 이용한 경우
arr_numpy = np.arange(10)
print('numpy의 ndarray 데이터')                  # numpy의 ndarray 데이터
print('arr_numpy: ', arr_numpy)                 # arr_numpy:  [0 1 2 3 4 5 6 7 8 9]

arr_numpy_view = arr_numpy[:]
arr_numpy_view[0] = 100

print('arr_numpy: ', arr_numpy)                 # arr_numpy:  [100   1   2   3   4   5   6   7   8   9]
                                                # 영향 O

                                           
# numpy의 ndarray에서 copy()를 이용한 경우
arr_numpy = np.arange(10)
print('numpy의 ndarray에서 copy() 사용한 경우')   # numpy의 ndarray에서 copy()를 이용한 경우
print('arr_numpy: ', arr_numpy)                 # arr_numpy:  [0 1 2 3 4 5 6 7 8 9]

arr_numpy_view = arr_numpy[:].copy()
arr_numpy_view[0] = 100

print('arr_numpy: ', arr_numpy)                 # arr_numpy:  [0 1 2 3 4 5 6 7 8 9]       
                                                # 영향 X                                     