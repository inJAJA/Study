"""
# 조건부 확률
"""
import enum, random

# Enum을 사용하면 각 항목에 특정 값을 부여할 수 있으며
# 파이썬 코드를 더욱 깔끔하게 만들어 준다.
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:                         # older_girl
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL: # both_girls
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:  # either_girl
        either_girl += 1

print("P(both | older): ", both_girls / older_girl )    # 0.5007089325501317
print("P(both | either): ", both_girls / either_girl )  # 0.3311897106109325

