# 특정 함수 f를 입력하면 f의 결과를 2배로 만들어 주는 함수를 반환하는 함수
def doubler(f):                    
    # f를 참조하는 새로운 함수
    def g(x):
        return 2 *f(x)
    return g
# 이 함수는 특별한 경우에만 작동함   


def f1(x):
    return x + 1
 
g = doubler(f1)                                   # g = doubler.g( )


assert g(3) == 8, "(3 + 1) * 2 should equal 8"
assert g(-1) == 0, "(-1 + 1 ) * 2 should equal 0"
# 두개 이상의 인자를 받는 함수의 경우 문제 발생

def f2(x, y):
    return x + y

g = doubler(f2)

try:
    g(1, 2)
except TypeError:
    print("as defined, g only takes one argument") #as defined, g only takes one argument


""" 임의의 수의 인자를 받는 함수 생성 : ( * )사용 """
def magic(*args, **kwargs):
    print("unnamed args: ", args)
    print("keyword args: ", kwargs)


magic(1, 2, key = "word", key2 ="word2")
# unnamed args:  (1, 2)
# keyword args:  {'key': 'word', 'key2': 'word2'}


# magic(key = "word", key2 ="word2", 1, 2)   
# => SyntaxError: positional argument follows keyword argument


magic(1, 2, 3, key = "word", key2 ="word2", key3 = "word3", key4 ="word4")
# unnamed args:  (1, 2, 3)
# keyword args:  {'key': 'word', 'key2': 'word2', 'key3': 'word3', 'key4': 'word4'}
"""
위의 함수에서
# args   : 이름이 없는 인자로 구성된 tuple
# kwargs : 이름이 주어진 인자로 구성된 dicktionary

: 정해진 수의 인자가 있는 함수를 호출할 때도 list나 dictionary로 인자 전달 할 수 있다.   
"""


def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z" : 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2+ 3 should be 6 "


def doubler_correct(f):
    """ f의 인자에 상관없이 작동한다. """
    def g(*args, **kwargs):
        """ g의 인자가 무엇이든 간에 f로 보내준다."""
        return 2 * f(*args, **kwargs)
    return g


g = doubler_correct(f2)
assert g(1, 2) == 6, "doubler should work now"

