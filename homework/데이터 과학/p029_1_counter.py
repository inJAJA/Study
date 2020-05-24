"""
# Counter
 : 연속된 값을 defaultdict(int)와 유사한 객체로 젼환해 준다.
   key와 value의 빈도를 연결시켜 줌
"""

from collections import Counter

c = Counter([0, 1, 2, 0])               # c = {0 : 2, 1 : 1, 2 : 1}


# document는 단어의 list 
word_counts = Counter(document)


""" most_common """
# 가장 자주 나오는 단어 10개와 이 단어들의 빈도수를 출력
for word, count in word_counts.most_common(10):
    print(word, count)