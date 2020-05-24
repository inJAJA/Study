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

print(friendships)
# {0: [1, 2],    1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3, 5], 
#  5: [4, 6, 7], 6: [5, 8],    7: [5, 8],    8: [6, 7, 9], 9: [8]}


def number_of_friends(user):
    """user의 친구는 몇 명일까?"""
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connection = sum(number_of_friends(user) for user in users)

print(total_connection)                              # 24

num_users = len(users)
avg_connection = total_connection / num_users

print(avg_connection)                                # 2.4

# (user_id, number_of_friends)로 구성된 리스트 생성
num_friends_by_id = [(user["id"], number_of_friends(user))for user in users]  

num_friends_by_id.sort(                              # 정렬해 보자
    key = lambda id_and_friends: id_and_friends[1],  # num_friends 기준으로 
    reverse= True)                                   # 제일 큰 숫자부터 제일 작은 숫자순으로

print(num_friends_by_id)
# (user_id, num_friends)쌍으로 구성 되어 있다.
#[(1, 3), (2, 3), (3, 3), (5, 3), (8, 3),
# (0, 2), (4, 2), (6, 2), (7, 2), (9, 1)]


