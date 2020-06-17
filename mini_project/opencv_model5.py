# 완성작!
from keras.models import load_model
import numpy as np
import cv2

categories = ["carrot","chicken","egg","fish",'flour',"mashroom","meat", "onion", "paprika","potato"]

image = cv2.imread("D:/data/pred/pred.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('digit', bin_img)
cv2.waitKey(0)                                     

kernel = np.ones((3,3),np.uint8)

closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=4)

n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
#    n_labels : 라벨 번호, img_labeled : 각 레이블링 부분의 이미지 배열, lab_stats : 모두 레이블링 된 이미지 배열
print(n_labels)
print(labels.shape)
print(stats.shape)
print(centroids.shape)

ingredients = []

size_thresh = 5000
for i in range(1, n_labels):
    if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
        print(stats[i, cv2.CC_STAT_AREA])
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        img = image[y:y+h, x:x+w]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        model = load_model('D:/Study/mini_project/model_train2.h5') 

        img_digit = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)
                                                                

        img_digit = img_digit /255.0   

        img_input = img_digit.reshape(1, 256, 256, 3)   
        predictions = model.predict(img_input)  

        number = np.argmax(predictions) 
        label = categories[number]   
        print(label)

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 1) 

        location = (x + int(w *0.5), y -10)    
        font = cv2.FONT_HERSHEY_COMPLEX       # 중간크기 세리프 폰트
        fontScale = 1.2
        cv2.putText(image, str(label), location, font, fontScale, (0, 255, 0), 2)
        
        ingredients.append(label)
        

# for문 빠져나옴

cv2.imshow('result', image)  
cv2.waitKey(0)
print(ingredients)

np.save('./data/ingredient.npy', arr = ingredients)