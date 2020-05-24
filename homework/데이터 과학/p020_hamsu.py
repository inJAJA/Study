'''
함수 
: 0개 혹은 그 이상의 인자를 입력 받아 결과를 반환하는 규칙
: def 를 이용해 함수 정의
'''

def double(x):
    '''
    이곳은 함수에 대한 설명을 적어 놓는 공간
    예를 들어 '이 함수는 입력된 변수에 2를 곱한 값을 출력해 준다.'라는 설명 추가될 수 있음
    '''
    return x*2


# 파이썬 함수들은 변수로 할당되거나 함수의 인자로 전달할 수 있다는 점에서 
# 일금 함수(first-class)의 특성을 가짐
def apply_to_one(f):
    """인자가 1인 함수 f를 호출"""
    return f(1)

my_double = double                  # 방금 정의한 함수를 나타낸다.
x = apply_to_one(my_double)       
print(x)                            # 2


# lamda 함수 
y = apply_to_one(lambda x : x +4)   # 5
print(y)

another_double = lambda x : 2*x     # 이 방법은 최대한 피하도록 하자

def another_double(x):
    """대신 이렇게 작성하자."""
    return 2*x


"""함수의 인자에 기본값을 할당할 수 있다."""
def my_print(message = "my default message"):
    print(message)

my_print("hello")                   # "hello" 출력
my_print()                          # "my default message" 출력


def full_name(first = "what's-his-name", last = "something"):
    return print( first + " " + last )

full_name("Joel","Grus" )           # "Joel Grus" 출력
full_name("Joel")                   # "Joel something" 출력
full_name(last = "Grus")            # "what's-his-name Grus" 출력
