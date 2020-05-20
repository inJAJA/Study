# 함수의 가장 큰 목적 : 재사용

def sum1(a, b):          # 함수 정의 : sum = 함수명 / a, b = 두개의 값을 받아들이겠다.(변수)
    return a + b         # 되돌려준다.
    
a = 1
b = 2
c = sum1(a, b)

print(c)                 # 3

#### 곱셉, 나눗셈, 뺄셈 함수를 만드시오.
# mul1, div1, sub1
def mul1(a, b):
    return a * b

def div1(a, b):
    return a / b

def sub1(a, b):
    return a - b

a = 4
b = 2

print(mul1(a, b))
print(div1(a, b))
print(sub1(a, b))


def sayYeh():          # 매개변수 없어도 함수 만들 수 있다.
    return 'Hi'

aaa = sayYeh()
print(aaa)             # Hi


def sum1(a, b, c):     # 매개변수를 여러개 지정 가능 
    return a + b + c          
    
a = 1
b = 2
c = 34
d = sum1(a, b, c)

print(d)               # 37