import numpy as np
'''
# 집합함수
: 수학의 집합연산을 수행하는 함수
: 1d_array만을 대상으로 함
'''
arr1 =[2, 5, 7, 9, 5, 2]
arr2 = [2, 5, 8, 3, 1]
# np.unique()  : 배열요소에서 중복을 제거하고 결과 반환
new_arr1 = np.unique(arr1)
print(new_arr1)                                        # [2 5 7 9]

# np.union1d() : 배열 x와 y의 합집합을 정렬해서 반환
union = np.union1d(arr1, arr2)
print(union)                                           # [1 2 3 5 7 8 9]

# np.intersect1d() : 배열 x와 y의 교집합을 정렬해서 반환
intersect = np.intersect1d(arr1, arr2)
print(intersect)                                       # [2 5]

# np.setdiff1d(x, y) : 배열 x에서 ,y를 뺸 차집합을정렬해서 반환
setdiff = np.setdiff1d(arr1, arr2)
print(setdiff)                                         # [7 9]