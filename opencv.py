img_digit = cv.resize(img_digit, (28, 28), interpolation = cv.INTER_AREA)

img_digit = img_digit /255.0

img_input = img_digit.reshape(1, 28, 28, 1)
predictions = model.predict(img_input)  # 예측

number = np.argmax(predictions)    # 카테고리 출력
print(number)

cv.rectangle(img_color, (x, y), (x+w, y+h), (255, 255, 0), 2)

location = (x + int(w *0.5), y -10)
font = cv.FONT_HERSHEY_COMPLEX
fontScale = 1.2
cv.putText(img_color, str(number), location, font, fontScale, (0, 255, 0), 2)

cv.imshow('digit', img_digit)
cv.waitkey(0)

# for문 빠져나옴

cv.imshow('result', img_color)
cv.waitkey(0)