'''
# bool index 참조
: []안에 논리값(True / False)배열을 사용하여 요소를 추출하는 방법
: 논리값(bool)배열의 True에 해당하는 요소의 ndarray를 만들어 반환
'''
import numpy as np

arr = np.array([2, 4, 6, 7])
print(arr[np.array([True, True, True, False])])      # [2 4 6]