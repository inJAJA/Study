"""
# Tuple
  : 변경할 수 없는 list
  : list 에서 수정을 제외한 모든 기능을 튜플에 적용할 수 있다.
  : ( )를 사용해서 정의
"""
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4

# list는 수정 가능
my_list[1] = 3                       # [1, 3]

# tuple은 수정 불가능
try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")   # cannot modify a tuple


""" 함수에서 여러 값을 반환할 떄 tuple 사용하면 편함 """
def sum_and_product(x, y):
    return (x + y), (x * y)

sp = sum_and_product(2, 3)            # (5, 6)
s, p = sum_and_product(5, 10)         # s = 15, p =50


""" 다중 할당 (multiple assignment) """
x, y = 1, 2                           # x = 1, y = 2
x, y = y, x                           # x = 2, y = 1
 