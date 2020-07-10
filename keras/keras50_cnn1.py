from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D     # Convolution 2D(가로 * 세로) : 이미지 자르는 것
from keras.layers import Dense, Flatten


"""
# CNN
:이미지를 자른후, 여러 데이터를 이어붙여 data를 증폭시킨 후 연관성을 찾는다.
"""

model = Sequential() # 이미지를 가로, 세로 2, 2로 자르겠다.  
model.add(Conv2D(10, (2, 2), input_shape = (10, 10, 1)))                  # (N, 9, 9, 10)            
model.add(Conv2D(7, (3, 3)))                                              # (N, 7, 7, 7)
model.add(Conv2D(5, (2, 2), padding = 'same'))                            # (N, 7, 7, 5)
model.add(Conv2D(5, (2, 2)))                                              # (N, 6, 6, 5)
# model.add(Conv2D(5, (2, 2), strides = 2))                               # (N, 3, 3, 5)
# model.add(Conv2D(5, (2, 2), strides = 2, padding ='same'))              # (N, 3, 3, 5) : padding보다 strides가 우선
model.add(MaxPooling2D(pool_size = 2))                                    # (N, 3, 3, 5)
model.add(Flatten())                                                      # (N, 45)      : 2차원
model.add(Dense(1))                                                       # (N, 1)

model.summary()
"""
#Conv2D( 10,     (2, 2),     input_shape = (   10,    10,      1  ))
       filter  kernal_size 
               자르는 size  x = ( batch_size, heigth, width, channel )  4차원
                                 행(장수)     세로    가로     색깔   = 1 (흑백), 3(컬러)
                                                              색깔끼리 나눠준다.
# CNN_Output_Size
 : [(N - F + padding_size) / stride] + 1 
 :이 사이즈의 그림이 이전 filter 수많큼 생성된다.
                                                                         layer의 node수를 통해 data증폭
 ex) Conv2D(10, (3, 3), input_shape = (10, 10, 1)) => (None, 8, 8, 10) : (8 * 8) 이 10장 생성된다. (증폭)
     Conv2D( 7, (4, 4))                            => (None, 5, 5,  7) : (5 * 5) 이  7장 생성된다. (증폭)
     
# Conv2D만 사용하여 layer구성시 문제점
 : filter로 잘려져 중복된 부분이 훈련이 더 많이 된다. -> side는  훈련 1번             =>  상대적인 데이터 손실
                                                   -> 중첩부분 훈련 2번 -> 값이 치중됨

# padding
 : side를 '0'으로 채워 side data도 동일하게 훈련될 수 있도록 해준다. 
  -> 홀수로 채울시 padding을 입히는 위치는 머신이 정한다.                                                 
  = 'same' : kernal_size와 상관없이 input_shape와 동일한 shape로 전달해준다.
  = 'valid' : default, (유효한 영역만 출력, 따라서 출력 이미지 사이즈는 입력 사이즈보다 작다)

# strides
 : 필터의 이동 간격 
 : 가로, 세로 따로 지정 가능 => strides = (2,1) : 세로 2칸, 가로 1칸 (),[] 둘다 가능
 : 값이 커지면 data_size가 작아짐
 : default = 1 

# MaxPooling
 : 필요없는 쓰레기 값을 버리고 중요부분(가장 큰 값)만 가져온다.
 : 학습 parameter가 없다.
 ex) 4 * 4 image
     0  1  5  4        (2 * 2)
     0 15 27 26    =>  15  27
     0  4  7  8        5   13
     0  5  6 13
 
 : heigth(width) / pool_size = MaxPooling (나누고 난 후의 소수점은 버린다.)


# Flatten
 : data를 쫙 펴준다.
 : data를 2차원으로 변형시켜 Dense형을 이용하여 우리가 볼수 있는 숫자로 나오게 한다. 
 : 학습 parameter가 없다.

# CNN_parameter
 :  ( channel * kernal_size  * filter ) + ( bias * filter)
   = ( input_dim * kernal_size +  1 ) * output
 ex) (     1     *  (2 * 2)    +  1 ) *   10  

"""


