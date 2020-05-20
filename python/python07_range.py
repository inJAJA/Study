# range 함수(클래스)
a = range(10)
print(a)          # range(0, 10)
b = range(1, 11)
print(b)          # range(1, 11)

for i in a:
    print(i)      # 0 ~ 9
for i in b:
    print(i)      # 1 ~ 10

print(type(a))    # <class 'range'>

sum = 0
for i in range(1, 11):
    sum = sum + i
print(sum)