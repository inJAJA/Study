"""
# Generate
 : (주로 for문을 통해서) 반복할 수 있으며, generator의 각 항목은 필요한 순간에 그때그때 생성
 '함수'와 'yield'를 활용해 생성 가능  
  단점 : generator를 단 한번만 반복 가능
"""
def generate_range(n):
    i = 0
    while i < n:
        yield i             # yield가 호출될 때마다 generator에 해당 값 생성
        i +=1 

for i in generate_range(10):
    print(f"i : {i}")


def natural_numbers():
    """1, 2, 3, ...을 반환"""
    n = 1
    while True:
        yield n
        n += 1              # 무한 수열 생성


# 괄호 안에 for문 추가하여 generator만들기
evens_below_20 = [i for i in generate_range(20) if i % 2 == 0]
print(evens_below_20)       #[0, 2, 4, 6, 8, 10, 12, 14, 16, 18] 


# 실제 반복문이 시작되기 전까지는 generator가 생성되지 않는다.
data = natural_numbers()
evens = [x for x in data if x % 2 ==0]
even_squares = [x ** 2 for x in evens]
even_squared_ending_in_six = [x for x in even_squares if x % 10 ==6]
# # 등등


""" enumerate : (순서, 항목) 형태로 값을 반환 시킬 수 있다."""
names = ["Alice", "Bob", "Charlie","Debbie"]

# 파이썬스럽지 않다.
i = 0 
for i in range(len(names)):
    print(f"name {i} is {names[i]}")

# 파이썬스럽지 않다.
i = 0
for named in names:
    print(f"name {i} is {names[i]}")
    i += 1

# 파이썬스럽다.
for i, name in enumerate(names):
    print(f"name {i} is {names[i]}")

"""
name 0 is Alice
name 1 is Bob
name 2 is Charlie
name 3 is Debbie
"""