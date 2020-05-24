""" Python은 동적 타입(dynamically typed)언어다.
 : 변수를 올바르게만 사용한다면 변수의 type은 신경쓰지 않아도 된다."""

def add(a, b) :
    return a + b

assert add(10, 5) == 15                    # + is valid for numbers               
assert add([1, 2], [3]) == [1, 2, 3]       # + is valid for list
assert add("hi ", "there") == "hi there"    # + is valid for strings

try:
    add(10, "five")
except TypeError:
    print("cannot add an int to a string")


""" 정적 타입(statically typed)언어의 경우, 모든 함수나 객체의 type을 명시해야 함 """
def add(a : int, b :int) -> int:
    return a + b


add(10, 5)                                  # 정상 작동
add("hi", "there")                          # 작동 안함

