## dateutil
# datetime.strptime() 클래스 메서드를 사용할 때는 문자열에 맞는 형식 문자열 사용자가 제공해야함
# dateutil 패키지의 parse명령을 쓰면 자동으로 형식 문자열을 찾아 datetime클래스 객체를 만들어 준다.
import datetime
from dateutil.parser import parse
print(parse('2016-04-16'))                        # 2016-04-16 00:00:00

print(parse('Apr 16, 2016 04:05:32 PM'))          # 2016-04-16 16:05:32

# 다만 '월'과 '일'이 모두 12보다 작은 숫자일 때는 
# 먼저 나오는 숫자를 '월'로 
# 나중에 나오는 숫자를 '일'로 판단한다.
print(parse('6/7/2016'))                          # 2016-06-07 00:00:00


# timedelta 클래스
# 날짜나 시간의 간격을 구할 때는 두 개의 datetime.datetime 클래스 객체의 차이를 구한다.
# 결과는 datetime.timedelta클래스 객체로 반환된다.
date1 = datetime.datetime(2016, 2, 19, 14)
date2 = datetime.datetime(2016, 1, 2, 13)
td = date1 - date2
print(td)                                         # 48 days, 1:00:00             


# 속성
# days : 일수
# seconds : 초(0 ~ 86399)
# microseconds : 마이크로초 ( 0 and 999999 )
print(td.days)                                     # 48   
print(td.seconds)                                  # 3600  = 1:00:00
print(td.microseconds)                             # 0

# method:
# total_seconds() : 모든 속성을 초단위로 모아서 변환
print(td.total_seconds())                          # 4150800.0


# datetime.datetime클래스 객체에 datetime.timedelta클래스 객체를 더해서 새로운 시간 구할 수 있음
t0 = datetime.datetime(2018, 9, 1, 13)
print(t0)                                          # 2018-09-01 13:00:00

d = datetime.timedelta(days = 90, seconds = 3600)
print(d)                                           # 90 days, 1:00:00
print(t0 + d)                                      # 2018-11-30 14:00:00