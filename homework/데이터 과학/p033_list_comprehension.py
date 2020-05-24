"""
# list_comprehension
 : 기존의 list에서 특정 항목을 선택하거나 변환시킨 결과를 새로운 list에 저장하는 경우 사용
"""
even_numbers = [x for x in range(5) if x % 2 ==0]  # [0, 2, 4]
squares      = [x*x for x in range(5)]             # [0, 1, 4, 9, 16]
even_squares = [x*x for x in even_numbers]         # [0, 4, 16]


square_dict = {x : x*x for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16} 
square_set = {x*x for x in [1, -1]}        # {1}


# list에서 불필요한 값은 밑줄로 표기
zeros = [0 for _ in even_numbers]             
print(zeros)                               # [0, 0, 0] : even_numbers와 동일한 길이            


# 여러 for를 포함할 수 있다.
pairs = [(x, y)
          for x in range(10)
          for y in range(10)]                 
print(pairs)                               # [(0,0),(0,1),(0,2)...(9,8),(9,9)] 100개


increasing_pairs = [(x,y)                  # x < y인 경우만 해당
                     for x in range(10)    
                     for y in range(x+1, 10)]
print(increasing_pairs)                    # [(0,1),(0,2)..(1,2)..(7,8),(7,9),(8,9)]     