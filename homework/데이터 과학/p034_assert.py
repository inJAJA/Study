"""
# assert
 : 지정된 조건이 충족되지 않는다면 AssertionError 반환
"""
assert 1 + 1 ==2
assert 1 + 1 ==2, "1 + 1 should equal 2 nut didn't"
# 조건이 충족되지 않을 때 문구가 출쳑됌


def smallest_item(xs):
    return(xs)

# assert smallest_item([10, 20, 5, 40]) == 5       # AssertionError
# assert smallest_item([1, 0, -1, 2]) == -1        # AssertionError


def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)