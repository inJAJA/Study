import cv2 
import numpy as np
from keras.models import load_model

img_color = cv2.imread('./data/pred/onion.jpg', cv2.IMREAD_COLOR) 
img_gray = cv2.cvtColor( img_color, cv2.COLOR_BGR2GRAY)  

img_blurred = cv2.GaussianBlur(gray, ksize = (5, 5), sigmx = 0)   # 가우시안 블러 : 이미지를 흐릿하게 만들어 노이즈 줄임

ret, img_binary = cv2.adaptiveThreshold(img_bluerred, maxValue = 255, 
                                        cv2.THRESH_BINARY_INV | cv2.ADAPTIVE_THRESH_GAUSSIAN_C , 
                                       block_size = 19, C = 9) 

kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5, 5))        
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
'''
cv2.imshow('digit', img_binary)
cv2.waitKey(0)                                     
'''
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_color, contours, -1, (0, 255, 0), 1) # -1 = 모든 컨투어 그림/ 1 = 선의 두께

tenp_result = np.zeros((height, width, channel), dtype = np.uint8)

contours_dict = []

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)   
    
    cv2.rectangle(temp_result, (x, y), (x+w, y_h), color = (255, 255, 255), 2 )

    contours_dict.append({
        'contour' : contour,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'cx': x+(w/2),  # 컨투어를 감싼 중심 좌표
        'cy': y+(h/2)
    })

    MIN_AREA = 80,
    MIN_WIDTH, MIN_HEIGHT = 2.8,
    MIN_RATIO, MAX_RATIO = 0.25, 1.0
    
    possible_contours = []

    cnt = 0
    for d in contoudict:
        area = d['w']*d['h']
        ratio = d['w']/d['h']

    model = load_model('./mini_project/graph/model1.h5') 

    img_digit = cv2.resize(img_digit, (256, 256), interpolation = cv2.INTER_LINEAR)
                                                                

    img_digit = cv2.cvtColor( img_digit, cv2.COLOR_GRAY2BGR)

    img_digit = img_digit /255.0   

    img_input = img_digit.reshape(1, 256, 256, 3)   
    predictions = model.predict(img_input)  

    number = np.argmax(predictions)    
    print(number)

    cv2.rectangle(img_color, (x, y), (x+w, y+h), (255, 255, 0), 2) 

    location = (x + int(w *0.5), y -10)    
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1.2
    cv2.putText(img_color, str(number), location, font, fontScale, (0, 255, 0), 2)

    cv2.imshow('digit', img_digit) 
    cv2.waitKey(0)

# for문 빠져나옴

cv2.imshow('result', img_color)  
cv2.waitKey(0)