""" data 시각화 
# .pyplot   : 시각화를 단계별로 간편하게 만들 수 있는 구조 
# savefig() : 그래프 저장
# show()    : 화면에 띄우기
"""
from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp   = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]


# x축에 year, y축에 GDP가 있는 선 그래프 생성
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')


# 제목 더하기
plt.title("Nominal GDP")


# y축에 레이블을 추가
plt.ylabel("Billions of $")

# 그래프 보여주기
plt.show()


"""
# matplotlib의 주요 색상

문자	색상
-----------------
b	blue(파란색)
g	green(녹색)
r	red(빨간색)
c	cyan(청록색)
m	magenta(마젠타색)
y	yellow(노란색)
k	black(검은색)
w	white(흰색)

========================

# matplotlib의 주요 마커

마커	의미
------------------------
o	circle(원)
v	triangle_down(역 삼각형)
^	triangle_up(삼각형)
s	square(네모)
+	plus(플러스)
.	point(점)
"""