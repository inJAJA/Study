"""
# Sort
 : list를 자동으로 정렬해줌
 이미 만든 list를 망치고 싶지 않다면 sorted를 함수로 사용해서 새롭게 정렬된 list생성 가능
 '오름 차순'으로 정렬
"""
x = [4, 1, 2, 3]
y = sorted(x)           # y = [1, 2, 3, 4] / x = [4, 1, 2, 3]
x.sort()                # x = [1, 2, 3, 4]


""" 내림 차순 : reverse = True """
a = sorted([-4, 1, -2, 3], reverse= True)  
print(a)                # [3, 1, -2, -4]


""" 절댓값 : key=abs """
b = sorted([-4, 1, -2, 3], key=abs, reverse = True)  
print(b)                # [-4, 3, -2, 1]


# 빈도의 내림차순으로 단어와 빈도를 정렬
# wc = sorted(word_counts.items(),
#             key = lambda word_and_count : word_and_count[1],
#             reverse = True)
# print(wc)