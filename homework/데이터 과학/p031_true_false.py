"""
불(boolean) 타입이 존재하는 데 , 항상 대문자로 시작한다.
"""

one_is_less_than_two = 1 < 2             # True
true_equals_false = True == False        # False

""" 존재하지 않는 값은 None으로 표기 """
x = None
assert x == None, "this is the not Pythonic way to check for None"
assert x is None, "this is the Pythonic way to check for None"

""" 모두 거짓(False)
 Flase
 None
 []   (빈 list)
 {}   (빈 dict)
 ""
 set()
 0
 0.0
"""

""" and """
s = "string"
first_char = s and s[0]     # s = True : s[] 반환 / s = False : s 반환 
print(first_char)           # s


""" or """
x = None                    # x = 숫자, None이면
safe_x = x or 0             # 항상 숫자
safe_x = x if x is not None else 0


""" all : 모든 항목이 True면 True 반환 """
all([True, 1, {3}])          # True
all([True, 1, {}])           # False


""" any : 하나의 항목이라도 True면 True 반환 """
any([True, 1, {}])           # True


all([])                      # False
any([])                      # True
 