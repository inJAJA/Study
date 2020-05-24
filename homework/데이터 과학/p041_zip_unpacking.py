"""
# zip
 : 여러개의 list를 서로 상응하는 항목의 tuple로 구성된 list로 변환해줌
"""

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

# 실제 반복문이 시작되기 전까지는 묶어 주지 않는다.
pair  = [pair for pair in zip(list1, list2)]
print(pair)                                  # [('a', 1), ('b', 2), ('c', 3)]


""" 주어진 list의 길이가 다를 경우 :첫번쨰 list가 끝나면 멈춤  """


""" unpacking ( * )  : zip된 항목들을 zip함수에 개별적인 인자로 전달해 줌 """
pairs = [('a', 1), ('b', 2), ('c', 3)]

letters, numbers = zip ( *pairs )
print(letters, numbers)                       # ('a', 'b', 'c') (1, 2, 3) 

letters, numbers = zip(('a', 1), ('b', 2), ('c', 3))
print(letters, numbers)                       # ('a', 'b', 'c') (1, 2, 3)


def add(a, b): 
    return a + b

add(1, 2)                               # 3 

try:
    add([1, 2])
except TypeError:
    print("add expects two imputs")     # add expects two imputs

add(*[1, 2])                            # * : list안의 인자를 개별적으로 만들어줌 1, 2
                                        # 3
