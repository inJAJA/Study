import cv2 
import numpy as np
from keras.models import load_model

img_color = cv2.imread('./data/pred/pred.jpg', cv2.IMREAD_COLOR) 
img_gray = cv2.cvtColor( img_color, cv2.COLOR_BGR2GRAY)  


ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU ) 

kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (15, 15))        
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

cv2.imshow('digit', img_binary)
cv2.waitKey(0)                                     

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
'''
for i in contours:
    # cv2.drawContours(img_color, [contours[i]], -1, (0, 255, 0), 1) # -1 = 모든 컨투어 그림/ 1 = 선의 두께

    check = cv2.isContourConvex(contours[i])

    if not check:
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(img_color, [hull], -1, (0, 255, 0), 1)
        cv2.imshow('convexhull', img_color)
'''
for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)   

    length = max(w, h) + 60                 
    img_digit = np.zeros((length, length, 1), np.uint8) 
                                            

    new_x, new_y = x - (length - w)//2, y-(length - h)//2 

    img_digit = img_binary[new_y : new_y + length, new_x:new_x + length]  

    kernel = np.ones((5, 5), np.uint8)   
    img_digit = cv2.morphologyEx(img_digit, cv2.MORPH_DILATE, kernel) 

    cv2.imshow('digit', img_digit)
    cv2.waitKey(0)                     

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