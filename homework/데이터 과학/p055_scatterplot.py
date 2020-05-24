"""
# 산점도( scatterplot ) : plt.scatter()
: 두 변수 간의 연관 관계를 보여 주고 싶을 때 적합한 그래프
"""
friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels  = ['a','b','c','d','e','f','g','h','i']

from matplotlib import pyplot as plt

plt.scatter(friends, minutes)

# 각 포인트에 레이블을 달자
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
        xy= (friend_count, minute_count),     # 레이블을 데이터 포인트 근처에 두되
        xytext =(5, -5),                      # 약간 떨어져 있게 하자
        textcoords = 'offset points')


plt.title("Daily Minutes vs. Number of Friends")  # 제목
plt.xlabel("# of friends")                        # x축 레이블
plt.ylabel("daily minutes spent on the site")     # y축 레이블

plt.show()

"""
matplotlib이 자동으로 축의 범위를 설정하게 하면 
공정한 비교를 할 수 없는 산점도가 나올 수 있다.
"""
# 축 간 공정한 비교를 할 수 없는 산점도
test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")

plt.xlabel("test_1_grade")
plt.ylabel("test_2_grade")

# plt.show()


""" plt.axis("equal") """
# 공정한 비교를 할 수 있음
plt.axis("equal")

plt.show()