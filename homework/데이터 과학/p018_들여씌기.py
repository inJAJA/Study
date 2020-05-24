# '#'기호는 주석을 의미한다.
# 파이썬에서 주석은 실행되지 않지만, 코드를 이해하는 데 도움이 된다.
for i in [1, 2, 3, 4, 5]:
    print(i)                    # ' for i' 단락의 첫 번째 줄
    for j in [1, 2, 3, 4, 5]:
        print(j)                # ' for j' 단락의 첫 번째 줄 
        print(i + j)            # ' for j' 단락의 마지막 줄
    print(i)                    # ' for i' 단락의 마지막 줄
print("done looping")

long_winded_computation = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 
                          11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)
print(long_winded_computation)  # 210

# list
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

easier_to_read_list_of_lists = [[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]

# \ (backslash)
two_plus_three = 2 + \
                3               # \ 코드가 다음줄과 이어지는 것 명시 

print(two_plus_three)

for i in [1, 2, 3, 4, 5]:
    
    # 빈 줄이 있다는 것을 확인하자.
    print(i)