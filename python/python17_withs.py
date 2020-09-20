str = "this is dog img.jpg"

print(str.startswith('this'))   # True

print(str.endswith('jpg'))      # True

choice = ['apple', 'banana']

str = 'I like apple'

print(str.endswith(tuple(choice)))  # 튜플로 변환해 줘야 한다.

filename = ['apple.jpg', 'banana.png', 'orange.txt']
check = [name for name in filename if name.endswith(('.jpg', '.png'))]

print(check)    # ['apple.jpg', 'banana.png']
