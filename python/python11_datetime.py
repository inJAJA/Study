'''
# datetime package
1. datetime : 날짜, 시간 저장
2. date     : 날짜
3. time     : 시간
4. timedelta: 시간 구간 정보 저장
'''
from datetime import datetime

# .now()
# 현재 시각 출력
now = datetime.now()
print(now)                 # 2020-06-04 12:50:34.112665

# year
print(now.year)            # 2020

# month
print(now.month)           # 6
 
# day
print(now.day)             # 4

# hour
print(now.hour)            # 12

# minute
print(now.minute)          # 50

# second
print(now.second)          # 34

# microsecond
print(now.microsecond)     # 112665


# weekday()
# 요일 변환 : (0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일)
print(now.weekday())                                     # 3

# strftime()
# 문자열 반환
# now.strftime()

# date()
# 날짜 정보만 가지는 datetime.date클래스 객체 반환
print(now.date())                                         # 2020-06-04


# time()
# 시간 정보만 가지는 datetime.time 클래스 객체 반환
print(now.time())                                         # 14:15:40.988898


"""
# 날짜 및 시간 지정 문자열
 %Y : 앞의 빈자리를 0으로 채우는 4자리 '연도' 숫자
 %m : 앞의 빈자리를 0으로 채우는 2자리 '월' 숫자
 %d : 앞의 빈자리를 0으로 채우는 2자리 '일' 숫자
 %H : 앞의 빈자리를 0으로 채우는 24시간 형식 2자리 '시간'숫자
 %M : 앞의 빈자리를 0으로 채우는 2자리 '분' 숫자
 %S : 앞의 빈자리를 0으로 채우는 2자리 '초' 숫자
 %A : 영어로 된 '요일' 문자열
 %B : 영어로 된 '월' 문자열
"""

print(now.strftime("%A %d. %B %Y"))                       # Thursday 04. June 2020
print(now.strftime("%H시 %M분 %S초".encode('unicode-escape').decode()).encode().decode('unicode-escape'))
                                                          # 16시 40분 58초 / 한글일 때 인코딩 에러가 날때 해결법

# datetime.strptime()
# 첫번째 인수로는 날짜, 시간 정보를 가진 문자열
# 두번째 인수로는 그 문자열을 해독할 수 있는 형식 문자열을 넣는다.
print(datetime.strptime('2017-01-02 14:44', '%Y-%m-%d %H:%M'))



