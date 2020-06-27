# 같은 폴더 안에만 있으면 import로 불러 올 수 있다.
import p11_car
import p12_tv

print("====================")
print("do.py의 module 이름은", __name__)
print("====================")

p11_car.drive()
p12_tv.watch()

'''
운전하다.
car.py의 module 이름은  p11_car
시청하다
tv.py의 module 이름은  p12_tv      # 가져온 파일의 이름 = 파일명
====================
do.py의 module 이름은 __main__     # 실행시킨 시점의 이름 = __main__
====================
운전하다.
시청하다
'''