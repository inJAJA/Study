"""
# regex (regular expressions, 정규표현식)
 : regex을 사용하여 '문자열'을 찾을 수 있다.
"""
import re

re_example = [                             # 모두 True
    not re.match("a","cat"),               # 'cat'은 'a'로 시작하지 않기 때문에
    re.search("a","cat"),                  # 'cat' 안에는 'a'가 존재하기 때문에 
    not re.search("c","dog"),              # 'dog'안에는 'c'가 존재하지 않기 떄문에
    3 == len(re.split("[ab]","crabs")),    # a 혹은 b 기준으로 분리하면
                                           # ['c', 'r', 's']가 생성되기 때문에
    "R-D-" == re.sub("[0-9]","-","R2D2")   # 숫자를 " - " 로 대체하기 때문에 
]

print(all(re_example))                     # True
assert all(re_example), "all the regex examples should be True"  

"""
 re.match  : 문자열의 시작이 정규표현식과 같은지 비교

 re.search : 문자열에 정규표현식과 같은 부분이 있는지 찾음 
"""