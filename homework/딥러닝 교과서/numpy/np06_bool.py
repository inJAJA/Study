'''
# bool index 참조
: []안에 논리값(True / False)배열을 사용하여 요소를 추출하는 방법
: 논리값(bool)배열의 True에 해당하는 요소의 ndarray를 만들어 반환
'''
import numpy as np

arr = np.array([2, 4, 6, 7])
print(arr[np.array([True, True, True, False])])      # [2 4 6]

print(arr[arr % 3 == 1])                             # [4 7]   : 논리 연산의 결과가 True에 해닫되는 값 반환 


arr = np.array([2, 3, 4, 5, 6, 7])

print(arr % 2 == 0)                                  # [ True False  True False  True False] : 각 요소가 2로 나누어 떨어지는지
 
print(arr[arr % 2 == 0])                             # [2 4 6] : 각 요소 중 3로 나누어떨어지는 요소
