#2. 튜플
# 리스트와 거의 같으나, 삭제, 수정이 안된다.
# '고정 값'을 넣을 때 사용 가능
a = (1, 2, 3)
b = 1, 2, 3
print(type(a))       # <class 'tuple'>
print(type(b))       # <class 'tuple'>

# a.remove(2)
# print(a)           # AttributeError: 'tuple' object has no attribute 'remove'

print(a + b)         # (1, 2, 3, 1, 2, 3)
print(a * 3)         # (1, 2, 3, 1, 2, 3, 1, 2, 3)
                     # 객체 안만 삭제, 수정이 안됨 ( 연결은 가능 )

#print(a - 3)        # type error : 'tuple' - 'int'  
