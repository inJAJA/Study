"""
# class
 : class를 사용하여 객체 지향 프로그래밍(object_oriented programming)을 하면 
 데이터와 관련 함수응 하나로 묶어 줄 수 있다.
"""

# 예제 : 모임에 몇명 참가했는지 확인해 주는 CountingClicker 클래스 만들기
# count : 참석자 수(변수)
# cliker : count를 증가시킴
# read : 현재 count를 반환
# rest : count를 0으로 재설정

""" class 정의 : class뒤에 PascakCase로 class이름 표기 """
class CountingClicker:
    """ 함수처럼 클래스에도 주석을 추가할 수 있다.""" 
    def __init__(self, count =0):
        self.count = count

""" class는 0개 이상의 '멤버 함수'를 포함한다. 
    모든 '멤버 함수'의 첫 번째 인자는 해당 class의 instance를 의미하는 
    " self "로 정의 해야 한다.
"""
    

""" class의 이름으로 class의 instance를 생성 할 구 있다."""
clicker1 = CountingClicker()             # count = 0으로 생성된 instance
clicker2 = CountingClicker(100)          # count = 100으로 생성된 인스턴스
clicker3 = CountingClicker(count =100)   # 위와 동일

""" __repr__ : class instance를 문자열 형태로 변환해주는 dunber 메서드 """
def __repr__(self):
    return f"CountingClicker(count ={self.count}"


class CountingClicker:
    def __init__(self, count =0):
        self.count = count

# class를 활용할 수 있도록 public API만들기
    def click(self, num_times = 1):
    # 한번 실행할 때마다 num_times만큼 count가 증가
        self.count += num_times

    def read(self):
        return self.count

    def reset(self):
        self.count = 0


# assert를 사용하여 test조건 만들기
clicker = CountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"
clicker.click()
assert clicker.read() == 2, "after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, clicker should be back to 0"


# 부모 class의 모든 기능을 상속 받는 서브 클래스
class NoResetClicker(CountingClicker):
    #Counting Clicr와 동일한 메스드를 포함

    #하지만 reset 메서드는 아무런 기능이 없도록 변경된다.
    def reset(self):
        pass

clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.reset()
assert clicker2.read() == 1, "reset shouldn't do anything"