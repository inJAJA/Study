
num_friends = [100, 49, 41, 40, 25 ] #... 등등 더 많은 데이터

# 친구들의 수를 Counter와 plt.bar()를 이용해서 히스토그램으로 표현
from collections import Counter
import matplotlib.pyplot as plt

friend_counts = Counter(num_friends)
xs = range(101)                           # 최대값은 100
ys = [friend_counts[x] for x in xs]       # 히스토그램의 높이는 친구 수를 갖고있는 사용자 수

plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])

plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")

plt.show()


""" len() : 데이터 포인트의 개수"""
num_points = len(num_friends)              # 5


""" max(), min() : 쵀대/ 최솟값 """
largest_value   = max(num_friends)         # 100
smallest_value  = min(num_friends)         # 25


sorted_values = sorted(num_friends)        # [25, 40, 41, 49, 100]
smallest_value  = sorted_values[0]         # 25
second_smallest_value  = sorted_values[1]  # 40
second_largest_value  = sorted_values[-2]  # 49


