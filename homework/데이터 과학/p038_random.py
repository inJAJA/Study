""" random """
import random
random.seed(10)                      # 매번 동일한 결과를 변환해 주는 설정


""" random.random : 0 ~ 1 사이의 난수 생성 """
four_uniform_randoms = [random.random() for _ in range(4)]
print(four_uniform_randoms)
#[0.5714025946899135,           
# 0.4288890546751146,          
# 0.5780913011344704,          
# 0.20609823213950174]


""" random.seed : 고정된 난수 생성 """
random.seed(10)                      # seed를 10으로 설정
print(random.random())               # 0.5714025946899135
random.seed(10)                      # seed를 10으로 재설정 해도
print(random.random())               # 0.5714025946899135 재출력


""" random.randrange : rnage() 구간안에서 난수 생성 """
random.randrange(10)                 # range(10) = [0,1,2,3,4,5,6,7,8,9]에서 난수 생성
random.randrange(3, 6)               # rnage(3, 6) = [3, 4, 5] 에서 난수 생성


""" rnadom.shuffle : list의 하옴ㄱ을 임의 순서로 재정렬 """
up_to_ten = [1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10]
random.shuffle(up_to_ten)
print(up_to_ten)                     # [5, 6, 10, 2, 3, 8, 6, 7, 4, 1, 9] 


""" random.choice : list에서 임의의 항목 하나 선택 """
my_best_friend = random.choice(["Alice","Bob","Charlie"])
print(my_best_friend)                # Bob


""" random.sample : list에서 중복이 허용되지 않는 임의의 표본 list 만듦"""
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
print(winning_numbers)               # [4, 15, 47, 23, 2, 26]


""" 중복이 허용되는 임의의 표본 list 만들기 : random.choice 여러번 사용"""
four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)         # [2, 9, 5, 6]