# 자료형
#1. 리스트

a = [1, 2, 3, 4, 5]
b = [1, 2, 3, 'a', 'b'] # list에는 여러 자료형 넣을 수 있음/ BUT, numpy에서는 하나의 자료형만
print(b)

print(a[0] + a[3])     # 5
# print(b[0] + b[3])   # type error : 'int' + 'str'
print(type(a))         # <class 'list'>
print(a[-2])           # 4
print(a[1:3])          # 2, 3

a = [1, 2, 3, ['a', 'b', 'c']]
print(a[1])            # 2
print(a[-1])           # ['a', 'b', 'c']
print(a[-1][1])        # b                :  [큰 묶음][작은 묶음]


#1-2. 리스트 슬라이싱
a = [1, 2, 3, 4, 5]
print(a[:2])           # 1, 2


#1-3. 리스트 더하기
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)           # [1, 2, 3, 4, 5, 6]

c = [7, 8, 9, 10]
print(a + c)           # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(a * 3)           # [1, 2, 3, 1, 2, 3, 1, 2, 3]

# print(a + 2)         # type error : 'list' + 'int'    

# print(a[2] + 'hi')   # type error : 'int' + 'str'
print(str(a[2]) + 'hi')  # 3hi

f = '5'
# print(a[2] + f)      # type error : 'int' + 'str'  
print(a[2] + int(f))   # 8


######## 리스트 관련 함수 ########
#1. .apppend
a = [1, 2, 3]
a.append(4)            # a에다가 덧붙인다.
print(a)               # [1, 2, 3, 4]

# a = a.append(5)      # 문법 error : 다시 자기 자신한데 넣으면 안됌
# print(a)             # None

#2. .sort 
a = [1, 3, 4, 2]
a.sort()               # 차례대로 정렬
print(a)               # [1, 2, 3, 4]

#3. reverse
a.reverse()            # 뒤집다
print(a)               # [4, 3, 2, 1]

#4. index        
print(a.index(3))      # == a[3]
print(a.index(1))      # == a[1]

#5. insert
a.insert(0, 7)         # 0의 자리에 7을 삽입
print(a)               # [7, 4, 3, 2, 1]
a.insert(3, 3)
print(a)               # [7, 4, 3, 3, 2, 1]

#6. remove
a.remove(7)            # 7이라는 인자 값 삭제
print(a)               # [4, 3, 3, 2, 1]
a.remove(3)            # 먼저 걸린 인자만 지워짐
print(a)               # [4, 3, 2, 1]


# numpy 계산
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b
print(c)               # [5 7 9]

print(a * 3)           # [3 6 9] 