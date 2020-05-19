# 정수형
a = 1
b = 2
c = a + b
print(c)
d = a * b
print(d)
e = a / b
print(e)

# 실수형
a = 1.1
b= 2.2
c = a + b
print(c)     # 3.3000000000000003
             # 실수 값의 오차 생김
             # 부동 소수점 : 2진수로 표현하지 못하는 소수는 근사값으로 저장된다.
             #             ex) 0.3 => 0.01001100110011......(0011)의 무한 반복
             #                     =  0.010011001100110 로 저장
             # ' + ', ' - ' 연산시 오차가 생긴다.

d = a*b
print(d)                # 2.4200000000000004

e = a/b
print(e)                # 0.5

# 문자형
a = 'hel'
b = 'lo'
c = a+b
print(c)

# 문자 + 숫자
a = 123
b = '45'
# c = a+b                # 타입 에러 : 'int' + 'str'
# print(c)

# 숫자를 문자변환 + 문자
a = 123
a= str(a)                # 문자로 변환
print(a)
b = '45'
c = a+b
print(c)                 # 12345

# 문자를 숫자 변환 + 숫자
a = 123
b = '45'
b = int(b)
c = a+b 
print(c)                 # 168

# 문자열 연산하기
a = 'abcdefgh'
#    0,1,2,3,4,5,6,7
#   -8,-7,-6,-5,-4,-3,-2,-1
print(a[0])               # a
print(a[3])               # d
print(a[5])               # f
print(a[-1])              # h
print(a[-2])              # g
print(type(a)) # <class 'str'>

b = 'xyz'
print(a + b)              # abcdefghxyz

# 문자열 인덱싱 (slicing)
a = 'Hello, Deep learning'# 띄어 쓰기도 문자
#    01234567890123456789
#  - 09876543210987654321 
print(a[7])               # D
print(a[-1])              # g
print(a[-2])              # n            
print(a[3:9])             # lo, De                ( 3,  8)
print(a[3:-5])            # lo, Deep lea          ( 3, -6)
print(a[:-1])             # Hello, Deep learnin   ( 0, -2)
print(a[1:])              # ello, Deep learing    ( 1,   )
print(a[5:-4])            # , Deep lear           ( 5, -5)


