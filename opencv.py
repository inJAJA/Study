img_digit = cv.resize(img_digit, (28, 28), interpolation = cv.INTER_AREA)

img_digit = img_digit /255.0   # 이미지 픽셀 0~255

img_input = img_digit.reshape(1, 28, 28, 1)   # 학습때 요구된 input_shape
predictions = model.predict(img_input)  # 예측

number = np.argmax(predictions)    # 카테고리 출력(softmax)
print(number)

cv.rectangle(img_color, (x, y), (x+w, y+h), (255, 255, 0), 2) # 원본이미지 숫자마다 사각형 그려줌

location = (x + int(w *0.5), y -10)    # 이미지에 있는 숫자 위에 인식된 숫자를 적어줌
font = cv.FONT_HERSHEY_COMPLEX
fontScale = 1.2
cv.putText(img_color, str(number), location, font, fontScale, (0, 255, 0), 2)

cv.imshow('digit', img_digit) # 이미지에서 잘라낸 숫자부분을 가공한 결과 보여줌
cv.waitkey(0)

# for문 빠져나옴

cv.imshow('result', img_color)  # 원본이미지에 인식한 숫자를 보여줌
cv.waitkey(0)