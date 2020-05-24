"""
# Set
: 데이터 구조 중 유일한 항목의 집합을 나타내는 구조
 집합은 중괄호 { } 를 사용해서 정의한다.
"""

primes_below_10 = {2, 3, 5, 7}

s = set()                           # 비어있는 set 생성

""" .add : 추가 """
s.add(1)                            # { 1 } 
s.add(2)                            # { 1, 2 } 
s.add(2)                            # { 1, 2 } 

x = len(s)                          # 2
y = 2 in s                          # True
z = 3 in s                          # False


"""
특정 항목의 존재 여부 확인시 list보다 set를 사용하는 것이 효과적
 > 중복된 원소를 제거해 주기 때문
"""

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)          # 6

item_set = set(item_list)           # {1, 2, 3}
num_distinct_item = len(item_set)   # 3
distinct_item_list = list(item_set) # [1, 2, 3]   