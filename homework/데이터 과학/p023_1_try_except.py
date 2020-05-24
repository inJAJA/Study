"""
코드가 뭔가 잘못되면 파이썬은 예외(evception)가 발생했음을 알려준다.
예외를 제대로 처리해 주지 않으면 플로그램이 죽는데,
이를 방지해 주기 위해  try  ,  except  을 사용할 수 있다.
"""

try:
    print( 0 / 0)
except ZeroDivisionError:
    print("cannot divide by zero")