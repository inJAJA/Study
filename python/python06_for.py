a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

for i in a.keys():     # i에 a.key()의 값을 하나씩 집어넣는다.
    print(i)           # i = a[1]  : name
                       # i = a[2]  : phone
                       # i = a[3]  : birth

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:            # i에 a의 값을 하나씩 집어넣는다.
    i = i*i            
    print(i)
    print('melong')    # for문에 포함 O  (10번 출력) : 들여씌기 조심!
print('melong')        # for문에 포함 X  ( 1번 출력)

for i in a:
    print(i)


## while문
'''
while 조건문 :          # True인 동안 계속 돈다
    수행할 문장
'''

### if문
 
if 1 :                  #  만약 1이면
    print('Ture')
else :                  # 그렇지 않으면
    print('False')

if 3 :
    print('True')
else :
    print('Flase')

if 0 :                   # 0 = False, 1 ~ = True 
    print('True')
else :
    print('Flase')

if -1 :
    print('True')
else :
    print('Flase')

'''
비교 연산자

< , > , == , != , >= , <=

'''
a = 1
# if a = 1:              # a에 1대입
if a == 1 :              # a와 1이 같다
    print('출력잘돼')

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')

'''
조건 연산자

and, or, not

'''
money = 20000
card = 1
if money >= 30000 or card == 1 :
    print("한우먹자")
else:
    print('라면먹자')

jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:           # i 에 jumsu의 값을 하나씩 넣어준다.
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1
print(" 합격인원 :", number, "명")
# number  i   바뀐number 
#     0  90   1
#     1  25   1
#     1  67   2
#     2  45   2
#     2  80   3


# break, continue
print("========== break ==========")
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:           # i 에 jumsu의 값을 하나씩 넣어준다.
    if i < 30:
        print('break')
        break             # 그 문에서 제일 가까운 for문을 정지시킨다. 
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1
print(" 합격인원 :", number, "명")
# 경] 합격 [축
# break
#  합격인원 : 1 명

print("========== continue ==========")
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:           # i 에 jumsu의 값을 하나씩 넣어준다.
    if i < 60:
        print("continue")
        continue          # 하단 부분을 실행하지 않고 for문으로 돌아감
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1
print(" 합격인원 :", number, "명")
# 경] 합격 [축             # 90
# continue                # 25
# 경] 합격 [축             # 67
# continue                # 45
# 경] 합격 [축             # 80
#  합격인원 : 3 명