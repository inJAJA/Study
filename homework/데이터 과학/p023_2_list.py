"""
# list
 : 순서가 있는 자료의 집합(collection)
 : 배열 (array)와 유사하다. 
"""
integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_list = [integer_list, heterogeneous_list, []]

list_length = len(integer_list)
print(list_length)                   # 3

list_sum = sum(integer_list)
print(list_sum)                      # 6


""" 대괄호를 사용해 리스트의 n번째 값을 불러오거나 설정할 수 있다. """
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

zero = x[0]                          # 0 : 리스트의 순서는 0부터 시작한다.
one = x[1]                           # 1
nine = x[-1]                         # 9 : 리스트의 마지막 항목을 불러온다.
eight = x[-2]                        # 8 : 리스트의 뒤에서 두번째 항목을 불러온다.

x[0] = -1                            # 0번째 항목을 -1로 변경
print(x)                             # [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]


""" [1 : 9] 앞에꺼는 그대로 뒤에꺼는 -1 한다. """
first_three = x[:3]                  # 0 ~  2 : [-1, 1, 2]
three_to_end = x[3:]                 # 3 ~  9 : [ 3, 4, 5, 6, 7, 8, 9]
one_to_four = x[1:5]                 # 1 ~  4 : [ 1, 2, 3, 4]
last_three = x[-3:]                  # 7 ~  9 : [ 7, 8, 9]
without_first_and_last = x[1:-1]     # 1 ~ -2 : [ 1, 2, 3, 4, 5, 6, 7, 8]
copy_of_x = x[:]                     # 0 ~  9 : [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]

 
""" 간격(stride)을 설정하여 리스트 분리 """
every_third = x[::3]                 # 간격 3 : [-1, 3, 6, 9]
five_to_three = x[5:2:-1]            # 5 ~ 3  : [5, 4, 3]


""" in 연산자를 사용하여 리스트 안에서 항목의 존재 여부 확인"""
1 in [1, 2, 3]                       # True (참)
0 in [1, 2, 3]                       # False(거짓)


""" extend : 다른 리스트를 추가 """
x = [1, 2, 3]
x.extend([4, 5, 6])                  # x = [1, 2, 3, 4, 5, 6]

x = [1, 2, 3]
y = x + [4, 5, 6]                    # y = [1, 2, 3, 4, 5, 6]


""" . append : list에 항목 추가 """
x = [1, 2, 3]
x.append(0)                          # x = [1, 2, 3, 0]
y = x[-1]                            # 0
z = len(x)                           # 4

x, y = [1, 2]                        # x = 1, y = 2 

"""양쪽 항목의 개수가 다르면 ValueError가 발생
   ' _ ' 사용하여 버릴 항목 밑줄 표시 """
_, y = [1, 2]                        # y == 2 이고 첫번째 항목 무시    


