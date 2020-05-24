"""
# dict (dictionary, 사전)
  key(키) , value(값) 를 연결해 주어 빠르게 값을 검색할 수 있다. 
"""

empty_dict = {}                         # dictionary 생성
empty_dict2 = dict()                    # 잘 안사용함
grades = {"Joel": 80, "Tim": 95}        # key : Joel, Tim  / value : 80, 95 


""" key를 사용하여 value 불러오기 """
joel_grade = grades["Joel"]             # 80


""" 존재하지 않는 key를 입력하면 Keyerror 발생 """
try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade fot Kate!")         # no grade fot Kate


""" in :  key의 존재 여부 확인 가능 """
joel_has_grade = "Joel" in grades       # True  (참)
kate_has_grade = "Kate" in grades       # Flase (거짓)


""" .get : 입력한 key가 dick에 없어도 기본값 반환(error 안남)"""
joels_grade = grades.get("Joel", 0)     # 80
kates_grade = grades.get("kate", 0)     # 0    (기본값)
no_ones_grade = grades.get("No One", 0) # None (default)


""" [ ] : key와 value 수정 """
grades["Tim"] = 99                      # 80 -> 99
grades["kate"] = 100                    # {"Joel": 80, "Tim": 95, "Kate" : 100}
num_students = len(grades)              # 3


# 정형화된 데이터를 간단하게 나타낼때 주로 사용
tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data","#science","#datascience","#awsome","#yolo"]
}


""" .keys() : key에 대한 list """
tweet_keys = tweet.keys()               # dict_keys(['user', 'text', 'retweet_count', 
                                        #            'hashtags'])


""" .values() : value에 대한 list """
tweet_values = tweet.values()           # dict_values(['joelgrus', 'Data Science is Awesome', 
                                        #               100,  
                                        # ["#data","#science","#datascience","#awsome","#yolo"]]) 


""" .items() : (key, value)에 대한 list """
tweet_items = tweet.items()            
print(tweet_items)    
# dict_items([('user', 'joelgrus'), ('text', 'Data Science is Awesome'), 
#             ('retweet_count', 100), 
#             ('hashtags', ['#data', '#science', '#datascience', '#awsome', '#yolo'])])    


"user" in tweet.keys()                  # Ture : list에서 in을 사용하기 때문에 느리다.
"user" in tweet                         # Ture : dict에서 사용하기 때문에 빠르다.
"joelgrus" in tweet_values              # True



"""
dictionary의 key는 수정할 수 없다
lsit를 key로 사용할 수 없다.
"""


