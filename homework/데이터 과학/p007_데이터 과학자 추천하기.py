users= [{"id": 0, "name" : "hHro"},{"id": 1, "name" : "Dunn"},{"id": 2, "name" : "Sue"},{"id": 3, "name" : "Chi"},
        {"id": 4, "name" : "Thor"},{"id": 5, "name" : "clive"},{"id": 6, "name" : "Hicks"},{"id": 7, "name" : "Devin"},
        {"id": 8, "name" : "Kate"},{"id": 9, "name" : "Klein"}]

friendship_pairs = [(0, 1),(0, 2),(1, 2),(1, 3),(2, 3),(3, 4),
                    (4, 5),(5, 6),(5, 7),(6, 8),(7, 8),(8, 9)]

# 사용자별로 비어 있는 친구 목록 리스트를 지정하여 딕셔너리를 초기화
friendships = {user["id"] : [] for user in users}

print(friendships)
# {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}


# friendship_pairs 내 쌍을 차례대로 살펴보면서 딕셔너리 안에 추가
for i, j in friendship_pairs:
    friendships[i].append(j)    # j를 사용자 i를 친구로 추가
    friendships[j].append(i)    # i를 사용자 j를 친구로 추가

""" p007 : 데이터 과학자 추천하기 """
def foaf_ids_bad(user):
    # "foaf"는 친구의 친구("friend of a friend")를 의미하는 약자다.
    return [foaf_id
            for friend_id in friendships[user["id"]]
               for foaf_id in friendships[friend_id]]
              

print(foaf_ids_bad(users[0]))
# "Hero"에 관한 [0, 2, 3, 0, 1, 3]

print(friendships[0])    # [1, 2]
print(friendships[1])    # [0, 2, 3]
print(friendships[2])    # [0, 1, 3]

from collections import Counter
""" 
Counter 
"""

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]       # 사용자의 친구 개개인에 대해
         for foaf_id in friendships[friend_id]      # 그들의 친구들을 세어 보고
          if foaf_id != user_id                     # 사용자 자신과
           and foaf_id not in friendships[user_id]  # 사용자의 친구는 제외
    )

print(friends_of_friends(users[3]))                 # Counter({0: 2, 5: 1})

intersets = [
    (0, "Hadoop"),(0, "Hadoop"),(0, "Hadoop"),(0, "Hadoop"),
    (0, "Hadoop"),(0, "Hadoop"),(0, "Hadoop"),
    (1, "NoSQL"),(1, "NoSQL"),(1, "NoSQL"),(1, "NoSQL"),
    (1, "NoSQL"),(2, "Python"),(2, "Python"),(2, "Python"),
    (2, "Python"),(2, "Python"),(2, "Python"),(3, "R"),(3, "Python"),
    (3, "R"),(3, "R"),(3, "R"),
    (4, "machine learning"),(4, "machine learning"),(4, "machine learning"),
    (4, "libsvm"),
]