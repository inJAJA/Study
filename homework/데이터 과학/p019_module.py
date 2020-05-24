'''
모듈을 불러올 떄는 import를 사용한다.
'''

""" regex (regular expression)"""
import re
my_regex = re.compile("[0-9] + ", re.I)   # re.(함수 or 상수) 를 붙여 사용
print(my_regex)

# import regex
# my_regex = regex.compile("[0-9] + ", regex.I)    
# print(my_regex)

'''
모듈의 별칭을 사용하는 방법은 다음과 같다.
'''
import matplotlib.pyplot as plt   # matplotlib의 별칭 = plt
# plt.plot(...)                   # 별칭으로 사용하는 법

'''
모듈 하나에서 특정 기능만 필요하다면 해당 기능만 명시해서 불러 올 수 있다.
'''
from collections import defaultdic, Counter
lookup = defaultdic(int)
my_counter = Counter()

'''
모듈의 기능들을 통쨰로 불러와서 기존의 변수들을 덮어쓰는 것으 좋지 않다.
'''
match =10
# from re import                     # 이런! re에도 match라는 함수가 존재한다.
print(match)                       # "function match "

