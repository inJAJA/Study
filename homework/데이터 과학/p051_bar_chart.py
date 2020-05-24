"""
# 막대 그래프 ( bar_chart ) : plt.bar()
 : 이산적인( discrete ) 항목들에 대한 변화를 보여 줄 때 사용하면 좋다.
"""
movies = ["Annie Hall","Ben-Hur","Casablanca","Gandhi","Weat Side Story"]
num_oscars = [5, 11, 3, 8, 10]

from matplotlib import pyplot as plt

# 막대의 x좌표는 [0, 1, 2, 3, 4], y좌표는 [num_oscars]로 설정
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")             # 제목 추가
plt.ylabel("# of Acodemy Awards")           # y축에 레이블 추가


# x축 각 막대의 중앙에 영화 제목을 레이블로 추가
plt.xticks(range(len(movies)), movies)

plt.show()


""" 히스토그램( histogram ) 
 : 정해진 구간에 해당되는 항목의 개수를 보여준다.
 : 값의 분폴르 관찰할 수 있는 그래프의 형태       """

from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]


# 점수는 10점 단위로 그룹화 한다. 100점은 90점대에 속한다.
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)
print(histogram)                  # Counter({80: 4, 90: 3, 70: 3, 0: 2, 60: 1})

                                           
                                            # 각 막대를 오른쪽으로 5 만큼 이동
plt.bar([x + 5 for x in histogram.keys()],  # 각 막대의 중심점을 5로 맞춰줌
        histogram.values(),                 # 각 막대의 높이 정함
        10,                                 # 너비는 10
        edgecolor = (0, 0, 0))              # 각 막대의 테두리는 검은색으로 설정           


plt.axis([-5, 105, 0, 5])                   # x축은 -5 부터 105 까지
                                            # y축은  0 부터   5 까지


plt.xticks([10 * i for i in range(11)])     # x축의 레이블은 0, 10, ..., 100
plt.xlabel("Decile")                        # x축에 레이블 추가
plt.ylabel("# of Students")                 # y축에 레이블 추가
plt.title("Distribution of Exam 1 Grades")  # 제목 추가

plt.show()


""" y 축을 0에서 시작하지 않으면 오해를 불러일으키기 쉽다."""
mentions =[500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")


# 이렇게 하지 않으면 matplotlib이 x축에 0, 1레이블을 달고
# 주변부 어딘가에 +2.013e3이라고 표기해 둘 것이다
plt.ticklabel_format(useOffset=False)


# 오해를 불러일으키는 y축은 500이상의 부분만 보여 줄 것이다,
# plt.axis([2016.5, 2018.5, 499, 506])
# plt.title("Look at the 'huge' Increase")
# plt.show()


# y축이 오해를 불러 일으키지 않는 그래프
plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge anymore")
plt.show()