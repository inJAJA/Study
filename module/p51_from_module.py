from machine.car import drive
from machine.tv import watch

drive()   # 운전하다
watch()   # 시청하다

from machine import car
from machine import tv

car.drive()
tv.watch()

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()
    
     # 폴더 경로
from machine.test import car
from machine.test import tv

drive()
watch()

from machine import test
from machine import tv

test.car.drive()
test.tv.watch()