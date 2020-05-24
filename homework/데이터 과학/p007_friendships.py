# 데이터
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

interests = [
    (0, "Hadoop"),(0, "Big Data"),(0, "HBase"),(0, "Java"),
    (0, "Spark"),(0, "Storm"),(0, "Cassandra"),
    (1, "NoSQL"),(1, "MongoDB"),(1, "Cassandra"),(1, "HBase"),
    (1, "Postgres"),(2, "Python"),(2, "scikit-learn"),(2, "scipy"),
    (2, "numpy"),(2, "statsmodels"),(2, "pandas"),(3, "R"),(3, "Python"),
    (3, "statistics"),(3, "regression"),(3, "probability"),
    (4, "machine learning"),(4, "regression"),(4, "decision trees"),
    (4, "libsvm"),(5, "Python"),(5, "R"),(5, "Java"),(5, "C++"),
    (5, "Haskell"),(5, "programming languages"),(6, "statistics"),
    (6, "probability"),(6, "mathematics"),(6, "theory"),
    (7, "machine learing"),(7, "scikit-learn"),(7, "Mahout"),
    (7, "nerual network"),(8, "nerual network"),(8, "deep learning"),
    (8, "Big Data"),(8, "artificial intelligence"),(9, "Hadoop"),
    (9, "Java"),(9, "MapReduce"),(9, "Big Data"),
]

def data_scientists_who_like(target_interest):
    """특정 관심사를 갖고 있는 모든 사용자 id를 반환해 보자."""
    return [user_id
            for user_id, user_interest in interests
             if user_interest == target_interest
            ]

from collections import defaultdict

# 키가 관심사, 값이 사용자 id
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append((user_id))

print(user_ids_by_interest)
# defaultdict(<class 'list'>, {'Hadoop': [0, 9], 'Big Data': [0, 8, 9], 
# 'HBase': [0, 1], 'Java': [0, 5, 9], 'Spark': [0], 'Storm': [0], 
# 'Cassandra': [0, 1], 'NoSQL': [1], 'MongoDB': [1], 'Postgres': [1], 
# 'Python': [2, 3, 5], 'scikit-learn': [2, 7], 'scipy': [2], 'numpy': [2], 
# 'statsmodels': [2], 'pandas': [2], 'R': [3, 5], 'statistics': [3, 6], 
# 'regression': [3, 4], 'probability': [3, 6], 'machine learning': [4], 
# 'decision trees': [4], 'libsvm': [4], 'C++': [5], 'Haskell': [5], 
# 'programming languages': [5], 'mathematics': [6], 'theory': [6], 
# 'machine learing': [7], 'Mahout': [7], 'nerual network': [7, 8], 
# 'deep learning': [8], 'artificial intelligence': [8], 'MapReduce': [9]})


# 키가 사용자 id, 값이 관심사
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

print(interests_by_user_id)
#defaultdict(<class 'list'>, 
# {0: ['Hadoop', 'Big Data', 'HBase', 'Java', 'Spark', 'Storm', 'Cassandra'], 
# 1: ['NoSQL', 'MongoDB', 'Cassandra', 'HBase', 'Postgres'],
# 2: ['Python', 'scikit-learn', 'scipy', 'numpy', 'statsmodels', 'pandas'], 
# 3: ['R', 'Python', 'statistics', 'regression', 'probability'], 
# 4: ['machine learning', 'regression', 'decision trees', 'libsvm'], 
# 5: ['Python', 'R', 'Java', 'C++', 'Haskell', 'programming languages'], 
# 6: ['statistics', 'probability', 'mathematics', 'theory'], 
# 7: ['machine learing', 'scikit-learn', 'Mahout', 'nerual network'], 
# 8: ['nerual network', 'deep learning', 'Big Data', 'artificial intelligence'], 
# 9: ['Hadoop', 'Java', 'MapReduce', 'Big Data']})    


def most_common_interests_with(user):
    return Counter(
        interested_user_id
        for interest in interests_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user["id"]
    )

print(most_common_interests_with(users[0]))
# Counter({9: 3, 1: 2, 8: 1, 5: 1})

