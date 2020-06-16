import cv2
import numpy as np
from keras.models import load_model

img_color = cv2.imread('./data/pred/pred.jpg', cv2.IMREAD_COLOR) 
img_gray = cv2.cvtColor( img_color, cv2.COLOR_BGR2GRAY)  

img_blurred = cv2.GaussianBlur(gray, ksize = (5, 5), sigmx = 0)   # 가우시안 블러 : 이미지를 흐릿하게 만들어 노이즈 줄임

ret, img_binary = cv2.adaptiveThreshold(img_bluerred, maxValue = 255, adaptiveMethod = cv2.THRESH_BINARY_INV,
                                        thresholdType = cv2.ADAPTIVE_THRESH_GAUSSIAN_C , 
                                        block_size = 19, C = 9) 

kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5, 5))        
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

cv2.imshow('digit', img_binary)
cv2.waitKey(0)                                     

def get_contours(name, small, pagemask, masktype):
    
    mask = get_mask(name, small, pagemask, masktype)

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.7, 'get_mask', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_out = []

    for contour in contours:

        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT*height):
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    if DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)

    return contours_out

get_contours(img)
