#3. 딕셔너리 # 중복 X
# {키 : 벨류}
# {key : value}

a = {1: 'hi', 2: 'hello'}
print(a)
print(a[1])                 # hi

b = {'hi' : 1, 'hello' : 2}
print(b['hello'])           # 2


# 딕셔너리 요소 삭제
del a[1]                    # delete : key 1
print(a)                    # {2: 'hello'}
del a[2]                    # delete : ket 2
print(a)                    # {}

a = {1:'a', 1:'b', 1:'c'}
print(a)                    # {1: 'c'}

b = {1:'a', 2:'a', 3:'a'}
print(b)                    # {1: 'a', 2: 'a', 3: 'a'}


a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())             # dict_keys(['name', 'phone', 'birth'])
print(a.values())           # dict_values(['yun', '010', '0511'])

print(type(a))              # <class 'dict'>

print(a.get('name'))        # yun
print(a['name'])            # yun
print(a.get('phone'))       # 010
print(a['phone'])           # 010

