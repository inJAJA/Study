'''
# ndarray
* 주의점 : 다른 변수에 그대로 대입한 경우 해당 변수의 값을 변경하면 원래 ndarray배열의 값도 변경된다.
         -> copy() : 두개의 변수를 별도로 만든다.
'''
import numpy as np

# ndarray를 그대로 arr2변수에 대입한 경우
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)                                       # [1 2 3 4 5]

arr2 = arr1
arr2[2] = 100 

print(arr1)                                       # [  1   2 100   4   5]
# arr2 변수를 변경하면 원래 변수(arr1)도 영향 받음


# ndarray를 copy()를 사용해서 arr2에 대입한 경우
arr1 = np.array([1, 2, 3, 4, 5]) 
print(arr1)                                        # [1 2 3 4 5]

arr2 = arr1.copy()
arr2[0] = 100

print(arr2)                                        # [100   2   3   4   5]
# arr2 변수를 변경해도 원래 변수(arr1)에 영향 X





