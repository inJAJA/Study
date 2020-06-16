import cv2 
import numpy as np
from keras.models import load_model

img_color = cv2.imread('./data/pred/pred.jpg', cv2.IMREAD_COLOR) # 컬러로 이미지 불러오기
img_gray = cv2.cvtColor( img_color, cv2.COLOR_BGR2GRAY)  # gray로 색상변경
# img_gray = img_color

ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU ) # 이미지 이진화

kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5, 5))        # 모폴로지 연산적용 : 이진화결과 생겼을지 모르는 빈 공백 메꿈
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
                                        # ersion -> dilation
cv2.imshow('digit', img_binary)
cv2.waitKey(0)                                      # inshow()함수를 붙잡아둠 : 0 or None = 무한정 기다림 / int = ms (1000 = 1초) 

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL,  # 숫자별로 분리하기 위해 컨투어 검출
                                      cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)   # 숫자별로 경계박스 구함

    length = max(w, h) + 60                  # 가로, 세로중 긴방향을 택한후 여분을 추가하여 한변의 크기를 정함
    img_digit = np.zeros((length, length, 1), np.uint8) # 잘라낸 숫자 이미지를 저장할 빈 이미지 생성
                                            # data를 int8형으로 변경/ 양수만 표현 가능(0 ~ 255)

    new_x, new_y = x - (length - w)//2, y-(length - h)//2 # 숫자가 이미지의 정중앙에 오도록 경계박스의 시작 위치 조정

    img_digit = img_binary[new_y : new_y + length, new_x:new_x + length]  # 바이너리 이미지에서 숫자영역을 가져와 저장

    kernel = np.ones((5, 5), np.uint8)    # 숫자가 잘 인식되도록 팽창 모폴로지 연산 적용
    img_digit = cv2.morphologyEx(img_digit, cv2.MORPH_DILATE, kernel) 

    cv2.imshow('digit', img_digit)
    cv2.waitKey(0)                     # imshow()함수를 볼수 있도록 붙잡아 둠

    model = load_model('./mini_project/graph/model1.h5') # 학습모델 불러오기

    img_digit = cv2.resize(img_digit, (256, 256), interpolation = cv2.INTER_LINEAR) # 학습모델에 맞는 모양으로 변환
                                                                 # 쌍선형 보간법

    img_digit = cv2.cvtColor( img_digit, cv2.COLOR_GRAY2BGR)

    img_digit = img_digit /255.0   # 이미지 픽셀 0~255

    img_input = img_digit.reshape(1, 256, 256, 3)   # 학습때 요구된 input_shape
    predictions = model.predict(img_input)  # 예측

    number = np.argmax(predictions)    # 카테고리 출력(softmax)
    print(number)

    cv2.rectangle(img_color, (x, y), (x+w, y+h), (255, 255, 0), 2) # 원본이미지 숫자마다 사각형 그려줌
                  # 이미지,   시작점, 종료점 좌표, 색상(blue, green, red), 선두께

    location = (x + int(w *0.5), y -10)    # 이미지에 있는 숫자 위에 인식된 숫자를 적어줌
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1.2
    cv2.putText(img_color, str(number), location, font, fontScale, (0, 255, 0), 2)

    cv2.imshow('digit', img_digit) # 이미지에서 잘라낸 숫자부분을 가공한 결과 보여줌
    cv2.waitKey(0)

# for문 빠져나옴

cv2.imshow('result', img_color)  # 원본이미지에 인식한 숫자를 보여줌
cv2.waitKey(0)