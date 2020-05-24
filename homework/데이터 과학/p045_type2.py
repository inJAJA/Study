# list type 명시
def total(xs : list) -> float:
    return sum(total)


""" Typing module 사용 """
from typing import List                 # L은 대문자인 것 유의


def total(xs: List[float]) -> float:    # xs는 float객체를 가지고 있는 list
    return sum(total)                   


# 이렇게 변수의 타입을 명시 할 수 있다.
# 하지만 x가 int라는 것이 자명하기 때문에 type을 명시할 필요가 없다.
x : int =5


values = []                            # type이 명확하지 않음
best_so_far = None                     # type이 명확하지 않음


""" 변수를 정의할 때 type에 대한 힌트를 추가할 수 있다."""
from typing import Optional

values : List[int] = []
best_so_far : Optional[float] = None   # float이나 None으로 타입 명시


# 여기서 명시하고 있는 type들은 너무 자명하여 굳이 명시할 필요는 없다.
from typing import Dict, Iterable, Tuple


# key는 strings , values는 int
counts: Dict[str, int] = {"data": 1, "science": 2}


# list와 generator는 모두 Iterable이다.
if lazy:
    evens : Iterable[int] = (x for x in range(10) if x % 2 == 0)
else:
    evens = [0, 2, 4, 6, 8]


# tuple안의 각 항목들의 type을 구체적으로 명시
triple: Tuple[int, float, int] = (10, 2.3, 5)


""" 일급 함수(first-class functions)에 대해서도 type명시 가능"""
from typing import Callable


# repeater 함수가 str와 int를 인자로 받고
# str을 반환해 준다는 것을 명시
def twice(repeater: Callable[[str, int], str], s : str) -> str :
    return repeater(s, 2)


def comma_repeater(s: str, n: int) -> str :
    n_copies = [s for _ in range(n)]
    return ', '.join(n_copies)


assert twice(comma_repeater, "type hints") == "type hints, type hints"


"""명시된 type자체도 python객체이기 때문에 변수로 선언할 수 있다."""
Number = int
Numbers = List[Number]

def total(xs: Numbers) -> Number:
    return sum(xs)