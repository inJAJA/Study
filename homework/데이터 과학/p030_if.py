if 1 > 2:
    message = "if only 1 were greater than two.."
elif 1 > 3:
    message = "elif stands for 'elif'" 
else:
    message = "when all fails use else (if you want to)" 


# 한 줄로 표현
x = 5
paroty = "even" if x % 2 == 0 else "odd"


x = 0
while x < 10:
    print(f"{x} is less than 10")    # f는 .format의 줄임말 = print("{0}".format(x))
    x += 1

for x in range(10):
    print(f"{x} is less than 10")


"""continue, break """
for x in range(10):
    if x == 3:
        continue                      # 다음 경우(순서)로 넘어간다.
    if x == 5:
        break                         # for문 전체를 끝냄
    print(x)                          # 0, 1, 2, 4