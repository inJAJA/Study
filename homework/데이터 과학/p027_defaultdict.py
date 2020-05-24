"""
# defaultdict
 : 존재하지 않는 key가 주어진다면 
  이 key와 인자에서 주어진 값으로 dict에 새로운 항목을 추가해 준다.
"""

from collections import defaultdict      # collection 모듈에서 불러오기

""" defaultdict(int)"""
# word_counts = defaultdict(int)           # int()는 0을 생성
# for word in document:             
#     word_counts[word] 

""" defaultdict(list) : x['key'].append('value')"""
dd_list = defaultdict(list)              # list() : 빈 list를 생성
dd_list[2].append(1)                     # dd_list =  {2: [1]}


""" defaultdict(dict) : x["key1"]["key1_2"] = "value" """
dd_dict = defaultdict(dict)              # dict() : 빈 dict 생성
dd_dict["Joel"]["City"] = "Seattle"      # dd_dict = {"Joel" : {"City" : Seattle}} 


""" defaultdict(lambda: [0, 0]) : x[key][] """ 
dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] =1                         # dd_pair = {2 : [0, 1]}
